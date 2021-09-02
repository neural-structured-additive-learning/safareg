fac_processor <- function(term, data, output_dim, param_nr){
  # factor_layer
  list(
    data_trafo = function() as.integer(data[extractvar(term)]),
    predict_trafo = function(newdata) as.integer(newdata[extractvar(term)]),
    input_dim = as.integer(extractlen(term, data)),
    layer = function(x, ...)
      return(tf$one_hot(tf$cast(x, dtype="int32"), 
                        depth = nlevels(data[[extractvar(term)]])) %>% 
               tf$keras$layers$Dense(
                 units = output_dim,
                 kernel_regularizer = tf$keras$regularizers$l2(l = extractval(term, "la")),
                 name = makelayername(term, 
                                      param_nr),
                 ...
               )),
    coef = function(weights) as.matrix(weights)
  )
}

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
                      name = makelayername(term, param_nr), units = units)
  }else if(length(nlev)==2){
    layer <- vvc_block(ncolNum, nlev[1], nlev[2], penalty = controls$sp_scale(data) * P, 
                       name = makelayername(term, param_nr), units = units)
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

