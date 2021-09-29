# tp and ttp have different positioning internally (b,a instead of a,b)
# as this results in a sparser design and penalty matrix
tp_layer = function(a, b, pen=NULL, name=NULL, units = 1, with_layer = TRUE) {
  x <- tf_row_tensor(b, a)
  if(with_layer) x <- x %>% layer_dense(units = units, activation = "linear", 
                                        name = name, use_bias = FALSE, 
                                        kernel_regularizer = pen)
  return(x)
}

ttp_layer = function(a, b, c, pen=NULL, name=NULL, units = 1, with_layer = TRUE) {
  
  x <- tf_row_tensor(tf_row_tensor(b, c), a) 
  if(with_layer) x <- x %>% layer_dense(units = units, activation = "linear", 
                                        name = name, use_bias = FALSE, 
                                        kernel_regularizer = pen)
  
  return(x)
}

pen_vc <- function(P, strength, nlev)
{
  python_path <- system.file("python", package = "deepregression")
  splines <- reticulate::import_from_path("psplines", path = python_path)
  return(splines$squaredPenaltyVC(P = as.matrix(P), strength = strength, nlev = nlev))
  
}

vc_block <- function(ncolNum, levFac, penalty = NULL, name = NULL, units = 1, with_layer = TRUE){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac)
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b = tf$one_hot(tf$cast(
      (tf_stride_cols(x, ncolNum+1)[,1]), 
      dtype="int32"), 
      depth=levFac)
    return(tp_layer(a, b, pen=penalty, name=name, units=units, with_layer=with_layer))
    
  }
  return(ret_fun)
}

vvc_block <- function(ncolNum, levFac1, levFac2, penalty = NULL, name = NULL, units = 1, with_layer = TRUE){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac1*levFac2)
  
  ret_fun <- function(x) ttp_layer(x[,as.integer(1:ncolNum)],
                                   tf$one_hot(tf$cast(x[,as.integer((ncolNum+1))], dtype="int32"), 
                                              depth=levFac1),
                                   tf$one_hot(tf$cast(x[,as.integer((ncolNum+2))], dtype="int32"), 
                                              depth=levFac2), 
                                   pen=penalty, name=name, units=units, with_layer=with_layer)
  
  return(ret_fun)
}

pen_vc_gen <- function()
{
  
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

factorization_machine <- function(embedding_dim,
                                  xlev,
                                  ylev,
                                  zlev) {
  
  dotprod <- function(a,b) tf$math$reduce_sum(tf$multiply(a,b), 
                                              axis=2L)
  
  dot_fun <- keras_model_custom(name = "factorization_machine", function(self) {
    self$x_embedding <-
      layer_embedding(
        input_dim = xlev + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = "first_level_embedding"
      )
    self$y_embedding <-
      layer_embedding(
        input_dim = ylev + 1,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = "second_level_embedding"
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
          name = "third_level_embedding"
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