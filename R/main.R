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
    data_trafo = function() lapply(data[extractvar(term)], int_0based),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], int_0based),
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
    data_trafo = function() lapply(data[extractvar(term)], int_0based),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], int_0based),
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

  vars <- extractvar(term)
  byt <- form2text(extractval(term, "by"))
  # extract gam part
  processor_name <- get_processor_name(term)
  gampart <- get_gam_part(term, wrapper=processor_name)
  if(length(setdiff(vars, c(extractvar(gampart), extractvar(byt))))>0)
    stop("vc terms currently only support one gam term with by term(s).")
  
  nlev <- sapply(data[extractvar(byt)], nlevels)
  if(any(nlev==0))
    stop("Can only deal with factor variables as by-terms in vc().")
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = gampart, 
                                        data = data, 
                                        controls = controls)
  
  ncolNum <- ncol(evaluated_gam_term[[1]]$X)
  
  sp_and_S <- extract_sp_S(evaluated_gam_term)
  P <- sp_and_S[[1]][[1]] * sp_and_S[[2]][[1]]
  
  if(length(nlev)==1){
    layer <- vc_block(ncolNum, nlev, penalty = controls$sp_scale(data) * P, 
                      name = makelayername(term, param_nr), units = output_dim,
                      with_layer = controls$with_layer)
  }else if(length(nlev)==2){
    layer <- vvc_block(ncolNum, nlev[1], nlev[2], penalty = controls$sp_scale(data) * P, 
                       name = makelayername(term, param_nr), units = output_dim,
                       with_layer = controls$with_layer)
  }else{
    stop("vc terms with more than 2 factors currently not supported.")
  }
  
  list(
    data_trafo = function() do.call("cbind", c(list(evaluated_gam_term[[1]]$X), 
                                               lapply(data[extractvar(byt)], int_0based))),
    predict_trafo = function(newdata) do.call("cbind", c(
      list(predict_gam_handler(evaluated_gam_term, newdata = newdata)),
      lapply(data[extractvar(byt)],int_0based))),
    input_dim = as.integer(ncolNum + length(nlev)),
    # plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

#' Processor for varying coefficient terms with one factor level using array mode
#' 
#' @export
#' 
am_processor <- function(term, data, output_dim, param_nr, controls){
  
  vars <- extractvar(term)
  byt <- form2text(extractval(term, "by"))
  # extract gam part
  processor_name <- get_processor_name(term)
  gampart <- get_gam_part(term, wrapper=processor_name)
  if(length(setdiff(vars, c(extractvar(gampart), extractvar(byt))))>0)
    stop("vc terms currently only support one gam term with by term(s).")
  
  nlev <- sapply(data[extractvar(byt)], nlevels)
  if(any(nlev==0))
    stop("Can only deal with factor variables as by-terms in vc().")
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = gampart, 
                                        data = data, 
                                        controls = controls)
  
  ncolNum <- ncol(evaluated_gam_term[[1]]$X)
  
  sp_and_S <- extract_sp_S(evaluated_gam_term)
  P <- sp_and_S[[1]][[1]] * sp_and_S[[2]][[1]]
  
  if(length(nlev)==1){
    layer <- arraym_block(ncolNum, nlev, penalty = controls$sp_scale(data) * P, 
                          name = makelayername(term, param_nr))
  #}else if(length(nlev)==2){
  }else{
    stop("am terms with more than 1 factor currently not supported.")
  }
  
  data_trafo = function() do.call("cbind", c(list(evaluated_gam_term[[1]]$X), 
                                             lapply(data[extractvar(byt)], int_0based)))
  predict_trafo = function(newdata) do.call("cbind", c(
    list(predict_gam_handler(evaluated_gam_term, newdata = newdata)),
    lapply(newdata[extractvar(byt)],int_0based)))
  
  list(
    data_trafo = data_trafo,
    predict_trafo = predict_trafo,
    plot_fun = function(self, weights, grid_length) 
      gam_plot_data(self, weights, grid_length, pe_fun = pe_fun_am),
    partial_effect = function(weights, newdata=NULL){
      if(is.null(newdata)){
        return(pe_fun_am(list(predict_trafo=function(df) data_trafo()),
                         as.data.frame(data[c(extractvar(gampart), extractvar(byt))]), 
                         weights))
      }else{
        return(pe_fun_am(list(predict_trafo=predict_trafo),
                              df = newdata, weights))
      }
    },
    get_org_values = function() data[c(extractvar(gampart), extractvar(byt))],
    input_dim = as.integer(ncolNum + length(nlev)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

pe_fun_am <- function(pp, df, weights){
  
  pmat <- pp$predict_trafo(df)
  rowSums((pmat[,1:(ncol(pmat)-1)]%*%weights) * model.matrix(~ -1 + ., data = df[,2,drop=F]))
  
}

#' Processor for factorization of two or three-level factor interactions
#' 
#' @export
#' 
fz_processor <- function(term, data, output_dim, param_nr, controls = NULL){
  # factor_layer
  dim <- extractval(term, "dim")
  if(is.null(dim)) dim <- 10
  trms <- extractvar(term)
  xlev <- nlevels(data[[trms[1]]])
  ylev <- nlevels(data[[trms[2]]])
  zlev <- NULL
  if(length(trms)==3) zlev <- nlevels(data[[trms[3]]])
  fz <- factorization(dim, xlev, ylev, zlev, 
                      name_prefix=makelayername(term, param_nr))
  
  layer = function(x, ...) fz(x)
  
  list(
    data_trafo = function() lapply(data[extractvar(term)], int_0based),
    predict_trafo = function(newdata) lapply(newdata[extractvar(term)], int_0based),
    input_dim = as.integer(extractlen(term, data)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}

#' Processor for varying coefficient terms with factorization of two factor levels
#' 
#' @export
#' 
vf_processor <- function(term, data, output_dim, param_nr, controls = NULL)
{
  vars <- extractvar(term)
  dim <- extractval(term, "dim")
  byt <- form2text(extractval(term, "by"))
  # extract gam part
  processor_name <- get_processor_name(term)
  gampart <- get_gam_part(term, wrapper=processor_name)
  if(length(setdiff(vars, c(extractvar(gampart), extractvar(byt))))>0)
    stop("vf terms currently only support one gam term and two by terms.")
  
  nlev <- sapply(data[extractvar(byt)], nlevels)
  if(any(nlev==0) | length(nlev)!=2)
    stop("Can only deal with (2) factor variables as by-terms in vf().")
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  evaluated_gam_term <- handle_gam_term(object = gampart, 
                                        data = data, 
                                        controls = controls)
  
  ncolNum <- ncol(evaluated_gam_term[[1]]$X)
  
  sp_and_S <- extract_sp_S(evaluated_gam_term)
  P <- sp_and_S[[1]][[1]] * sp_and_S[[2]][[1]]  
  
  layer <- vf_block(ncolNum, nlev[1], nlev[2], dim,
                    penalty = controls$sp_scale(data) * P, 
                    name = makelayername(term, param_nr), 
                    units = output_dim)
  
  list(
    data_trafo = function() do.call("cbind", c(list(evaluated_gam_term[[1]]$X), 
                                               lapply(data[extractvar(byt)], int_0based))),
    predict_trafo = function(newdata) do.call("cbind", c(
      list(predict_gam_handler(evaluated_gam_term, newdata = newdata)),
      lapply(data[extractvar(byt)],int_0based))),
    input_dim = as.integer(ncolNum + length(nlev)),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
  
}


#' Processor for additive factorization machine
#' 
#' @export
#' 
afm_processor <- function(term, data, output_dim, param_nr, controls){
  
  vars <- extractvar(term)
  nfac <- extractval(term, "fac")
  gamoptions <- "" # extractopts(term)
  # extract gam part
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  gamterms <- lapply(vars, function(var){
    
    gampart <- paste("s(", var, ")")
    warning("add gamoptions")
    
    evaluated_gam_term <- handle_gam_term(object = gampart, 
                                          data = data, 
                                          controls = controls)
    return(evaluated_gam_term)
    
  })

  input_dims <- sapply(gamterms, function(x) ncol(x[[1]]$X))
  penalties <- lapply(gamterms, function(x){
    
    sp_and_S <- extract_sp_S(x)
    
    sp_and_S <- list(sp = 1, 
                     S = list(do.call("+", lapply(1:length(sp_and_S[[2]]), function(i)
                       sp_and_S[[1]][[1]][i] * sp_and_S[[2]][[i]]))))
    return(
      as.matrix(bdiag(lapply(1:length(sp_and_S[[1]]), function(i) 
        controls$sp_scale(data) * sp_and_S[[1]][[i]] * sp_and_S[[2]][[i]])))
    )
    
  })
  
  splineparts <- lapply(1:nfac, function(j){
    
    lapply(1:length(gamterms), function(i){
      
      
      
      layer_spline(P = penalties[[i]], 
                   name = paste0(makelayername(term, param_nr),
                                 "te_", i, "_fac_", j), 
                   units = output_dim)
      
    })
    
  })
  
  layer = function(x, ...){
    
    xsplit <- tf$split(x, num_or_size_splits = as.integer(length(vars)), axis = 1L)
    
    splineparts <- lapply(1:nfac, function(j) lapply(1:length(vars), function(i)
      splineparts[[j]][[i]](xsplit[[i]])))
    
    firstpart <- lapply(1:nfac, function(j) tf$square(layer_add_identity(splineparts[[j]])))
    scndpart <- lapply(1:nfac, function(j) layer_add_identity(lapply(splineparts[[j]], tf$square)))
    combined <- lapply(1:nfac, function(j) tf$subtract(firstpart[[j]],scndpart[[j]]))
    
    res <- tf$multiply(0.5, layer_add_identity(combined))
    
    return(res)
  
  }
    
  data_trafo = function() do.call("cbind", lapply(gamterms, function(x) x[[1]]$X))
  predict_trafo = function(newdata) do.call("cbind", lapply(gamterms, function(x)
    predict_gam_handler(x, newdata = newdata)))
  
  list(
    data_trafo = data_trafo,
    predict_trafo = predict_trafo,
    get_org_values = function() data[vars],
    input_dim = as.integer(sum(input_dims)),
    layer = layer,
    partial_effect = function(weights, newdata=NULL){
      if(is.null(newdata)){
        return(rowSums(pe_fun_am(list(predict_trafo=function(df) data_trafo()),
                                 as.data.frame(data[c(extractvar(gampart), extractvar(byt))]), 
                                 weights))) 
      }else{
        return(rowSum(pe_fun_am(list(predict_trafo=predict_trafo),
                                df = newdata, weights)))
      }
    },
    coef = function(weights) as.matrix(weights)
  )
}