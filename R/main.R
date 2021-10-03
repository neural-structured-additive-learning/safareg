onehotfac <- function(x, lev){ 
  
  tf$squeeze(tf$one_hot(tf$cast(x, dtype="int32"), 
                        depth = lev), 
             axis=1L)
  
}
  
onehotia <- function(x, xlev, ylev){
  
  y = tf_stride_cols(x, 2L)
  x = tf_stride_cols(x, 1L)
  xoh = onehotfac(x, xlev)
  yoh = onehotfac(y, ylev)
  return(tf_row_tensor(xoh, yoh))
  
}

#' Processor for factor terms with many levels
#' 
#' @export
#' 
fac_processor <- function(term, data, output_dim, param_nr, controls = NULL){
  # factor_layer
  la <- extractval(term, "la", null_for_missing = TRUE)
  kr <- NULL
  if(!is.null(la))
    kr <- tf$keras$regularizers$l2(l = la)
  
  if(controls$with_layer){
    layer = function(x, ...)
      return(tf$keras$layers$Dense(
        units = output_dim,
        kernel_regularizer = kr,
        use_bias = FALSE,
        name = makelayername(term, 
                             param_nr),
        ...
      )(onehotfac(x, nlevels(data[[extractvar(term)]]))))
  }else{
    layer = function(x) onehotfac(x, nlevels(data[[extractvar(term)]]))
  }
    
  list(
    data_trafo = function() lapply(data[extractvar(term)], as.integer),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], as.integer),
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

#' Processor for interactions of factor terms with many levels
#' 
#' @export
#' 
interaction_processor <- function(term, data, output_dim, param_nr, controls = NULL){
  # factor_layer
  la <- extractval(term, "la", null_for_missing = TRUE)
  kr <- NULL
  if(!is.null(la))
    kr <- tf$keras$regularizers$l2(l = la)
  trms <- extractvar(term)
  xlev <- nlevels(data[[trms[1]]])
  ylev <- nlevels(data[[trms[2]]])
  
  if(controls$with_layer){
    layer = function(x, ...)
      return(tf$keras$layers$Dense(
        units = output_dim,
        kernel_regularizer = kr,
        use_bias = FALSE,
        name = makelayername(term, 
                             param_nr),
        ...
      )(onehotia(x, xlev = xlev, 
                 ylev = ylev)))
  }else{
    layer = function(x) onehotia(x, xlev = xlev, 
                                 ylev = ylev)
  }
  
  list(
    data_trafo = function() lapply(data[extractvar(term)], as.integer),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], as.integer),
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

#' Processor for varying coefficient terms with many factor levels
#' 
#' @export
#' 
vc_processor <- function(term, data, output_dim, param_nr, controls){
  # vc (old: vc, vcc)
  vars <- extractvar(term)
  byt <- form2text(extractval(term, "by"))
  # extract gam part
  gampart <- get_gam_part(term)
  if(length(setdiff(vars, c(extractvar(gampart), extractvar(byt))))>0)
    stop("vc terms currently only suppoert one gam term and one by term.")
  
  nlev <- sapply(data[extractvar(byt)], nlevels)
  if(any(nlev==0))
    stop("Can only deal with factor variables as by-terms in vc().")
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = gampart, 
                                        data = data, 
                                        controls = controls)
  
  ncolNum <- ncol(evaluated_gam_term[[1]]$X)
  
  P <- evaluated_gam_term[[1]]$S[[1]]
  
  if(length(nlev)==1){
    layer <- vc_block(ncolNum, nlev, penalty = controls$sp_scale(data) * P, 
                      name = makelayername(term, param_nr), units = units,
                      with_layer = controls$with_layer)
  }else if(length(nlev)==2){
    layer <- vvc_block(ncolNum, nlev[1], nlev[2], penalty = controls$sp_scale(data) * P, 
                       name = makelayername(term, param_nr), units = units,
                       with_layer = controls$with_layer)
  }else{
    stop("vc terms with more than 2 factors currently not supported.")
  }
  
  list(
    data_trafo = function() do.call("cbind", c(evaluated_gam_term[[1]]$X, 
                                               as.integer(data[byt]))),
    predict_trafo = function(newdata) do.call("cbind", c(
      predict_gam_handler(evaluated_gam_term, newdata = newdata),
      as.integer(data[byt]))),
    input_dim = as.integer(ncolNum + length(nlev)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

fm_processor <- function(term, data, output_dim, param_nr, controls = NULL){
  # factor_layer
  dim <- extractval(term, "dim")
  if(is.null(dim)) dim <- 10
  trms <- extractvar(term)
  xlev <- nlevels(data[[trms[1]]])
  ylev <- nlevels(data[[trms[2]]])
  zlev <- NULL
  if(length(trms)==3) zlev <- nlevels(data[[trms[3]]])
  fm <- factorization_machine(
    dim,
    xlev,
    ylev,
    zlev)
  
  layer = function(x, ...) fm(x)
  
  list(
    data_trafo = function() lapply(data[extractvar(term)], as.integer),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], as.integer),
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}