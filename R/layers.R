# tp and ttp have different positioning internally (b,a instead of a,b)
# as this results in a sparser design and penalty matrix
tp_layer = function(a, b, pen=NULL, name=NULL, units = 1L, with_layer = TRUE) {
  x <- tf_row_tensor(b, a)
  if(with_layer) x <- x %>% layer_dense(units = units, activation = "linear", 
                                        name = name, use_bias = FALSE, 
                                        kernel_regularizer = pen)
  return(x)
}

ttp_layer = function(a, b, c, pen=NULL, name=NULL, units = 1L, with_layer = TRUE) {
  
  x <- tf_row_tensor(tf_row_tensor(b, c), a) 
  if(with_layer) x <- x %>% layer_dense(units = units, activation = "linear", 
                                        name = name, use_bias = FALSE, 
                                        kernel_regularizer = pen)
  
  return(x)
}

layer_arram = function(a, b, pen=NULL, name=NULL) {
  
  lal <- linearArrayRWT(units = c(a$shape[[2]], b$shape[[2]]), P = pen, name = name)
  return(lal(list(a,b)))
  
}

pen_vc <- function(P, strength, nlev)
{
  python_path <- system.file("python", package = "safareg")
  layers <- reticulate::import_from_path("layers", path = python_path)
  return(layers$squaredPenaltyVC(P = as.matrix(P), strength = strength, nlev = nlev))
}

pen_simple <- function(P, strength)
{
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("psplines", path = python_path)
  return(layers$squaredPenalty(P = as.matrix(P), strength = strength))
}

linearArrayRWT <- function(units, P, name)
{
  python_path <- system.file("python", package = "safareg")
  layers <- reticulate::import_from_path("layers", path = python_path)
  return(layers$LinearArrayRWT(units = units, P = as.matrix(P), name = name))
}

vc_block <- function(ncolNum, levFac, penalty = NULL, name = NULL, units = 1, with_layer = TRUE){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac)
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b = tf$one_hot(tf$cast(
      tf_stride_cols(x, ncolNum+1), 
      dtype="int32"), 
      depth=levFac)
    if(length(b$shape)==2)
      b <- tf$squeeze(b, axis=1L)
    return(tp_layer(a, b, pen=penalty, name=name, units=units, with_layer=with_layer))
    
  }
  return(ret_fun)
}

arraym_block <- function(ncolNum, levFac, penalty = NULL, name = NULL){
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b = tf$one_hot(tf$cast(
      tf_stride_cols(x, ncolNum+1), 
      dtype="int32"), 
      depth=levFac)
    if(length(b$shape)==2)
      b <- tf$squeeze(b, axis=1L)
    return(layer_arram(a, b, pen=penalty, name=name))
    
  }
  return(ret_fun)
  
}

vf_block <- function(ncolNum, levFac1, levFac2, dim, penalty = NULL, name = NULL, units = 1){
  
  fz <- factorization(
    dim,
    levFac1,
    levFac2,
    NULL,
    name_prefix = name)
  
  if(!is.null(penalty))
    penalty = pen_simple(penalty, 1)
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b_c = tf$cast(tf_stride_cols(x,as.integer((ncolNum+1)),as.integer((ncolNum+2))), dtype="int32")
    fz_res = fz(b_c)
    
    basis_times_latent = tf$multiply(a, fz_res)
    ret <- basis_times_latent %>% layer_dense(units = units, activation = "linear", 
                                              name = name, use_bias = FALSE, 
                                              kernel_regularizer = penalty)
    
    return(ret)
    
    
  }
  
  return(ret_fun)
  
}

vvc_block <- function(ncolNum, levFac1, levFac2, penalty = NULL, name = NULL, units = 1, with_layer = TRUE){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac1*levFac2)
  
  b <- tf$one_hot(tf$cast(tf_stride_cols(x,as.integer((ncolNum+1))), dtype="int32"), 
                  depth=levFac1)
  c <- tf$one_hot(tf$cast(tf_stride_cols(x,as.integer((ncolNum+2))), dtype="int32"), 
                  depth=levFac2)
  
  if(length(b$shape)==2)
    b <- tf$squeeze(b, axis=1L)
  if(length(c$shape)==2)
    c <- tf$squeeze(c, axis=1L)
  
  ret_fun <- function(x) ttp_layer(x[,as.integer(1:ncolNum)], b, c, 
                                   pen=penalty, name=name, units=units, with_layer=with_layer)
  
  return(ret_fun)
}

layer_factor <- function(nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                         kernel_regularizer = NULL)
{
  
  ret_fun <- function(x) tf$one_hot(tf$cast(x[,1], dtype="int32"), depth = nlev) %>% layer_dense(
    units = units,
    activation = activation,
    use_bias = use_bias,
    name = name,
    kernel_regularizer = kernel_regularizer)
  return(ret_fun)
  
}

layer_random_effect <- function(freq, df)
{
  
  df_fun <- function(lam) sum((freq^2 + 2*freq*lam)/(freq+lam)^2)
  lambda = uniroot(function(x){df_fun(x)-df}, interval = c(0,1e15))$root
  nlev = length(freq)
  return(
    layer_factor(nlev = nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                 kernel_regularizer = regularizer_l2(l = lambda/sum(freq)))
  )
  
}

factorization <- function(embedding_dim, xlev, ylev, zlev, name_prefix) {
  
  dotprod <- function(a,b) tf$math$reduce_sum(tf$multiply(a,b), 
                                              axis=2L)
  
  dot_fun <- keras_model_custom(name = "factorization", function(self) {
    self$x_embedding <-
      layer_embedding(
        input_dim = xlev + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = paste0(name_prefix, "first_level_embedding")
      )
    self$y_embedding <-
      layer_embedding(
        input_dim = ylev + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = paste0(name_prefix, "second_level_embedding")
      )
    
    if(is.null(zlev)){
      self$dot <-
        layer_lambda(
          f = function(x) {
            dotprod(x[[1]], x[[2]])
          }
        )
      return(function(x, mask = NULL, training = FALSE) {
        y <- tf_stride_cols(x, 2L)
        x <- tf_stride_cols(x, 1L)
        x_embedding <- self$x_embedding(x)
        y_embedding <- self$y_embedding(y)
        self$dot(list(x_embedding, 
                      y_embedding))
      })
    }else{
      self$z_embedding <-
        layer_embedding(
          input_dim = zlev + 1,
          output_dim = embedding_dim,
          embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
          name = paste0(name_prefix, "third_level_embedding")
        )
      self$dot <-
        layer_lambda(
          f = function(x) {
            dotprod(dotprod(x[[1]], x[[2]]), x[[3]])
          }
  )
  return(function(x, mask = NULL, training = FALSE) {
    y <- tf_stride_cols(x, 2L)
    z <- tf_stride_cols(z, 3L)
    x <- tf_stride_cols(x, 1L)
    x_embedding <- self$x_embedding(x)
    y_embedding <- self$y_embedding(y)
    z_embedding <- self$z_embedding(activities)
    self$dot(list(x_embedding, 
                  y_embedding, 
                  z_embedding))
  })
    }
  })

return(dot_fun)

}
