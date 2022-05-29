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
#' @import deepregression
#' @export
#' 
fac_processor <- function(term, data, output_dim, param_nr, controls = NULL){
  # factor_layer
  la <- extractval(term, "la", default_for_missing = TRUE)
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
  la <- extractval(term, "la", default_for_missing = TRUE)
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
  ncolNum <- ncol(get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")())
  
  sp_and_S <- get_gamdata(gampart, param_nr, controls$gamdata, what="sp_and_S")
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
  
  data_trafo <- function() do.call("cbind", c(list(
    get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")()), 
    lapply(data[extractvar(byt)], int_0based)))
  predict_trafo <- function(newdata) do.call("cbind", c(
    list(get_gamdata(gampart, param_nr, controls$gamdata, what="predict_trafo")(newdata)),
    lapply(data[extractvar(byt)],int_0based)))
  
  data_trafo_red <- function() do.call("cbind", lapply(data[extractvar(byt)], int_0based))
  predict_trafo_red <- function(newdata) do.call("cbind", lapply(data[extractvar(byt)],int_0based))
  
  list(
    data_trafo = data_trafo_red,
    predict_trafo = predict_trafo_red,
    input_dim = as.integer(length(nlev)),
    # plot_fun = function(self, weights, grid_length) gam_plot_data(self, weights, grid_length),
    layer = layer,
    coef = function(weights) as.matrix(weights),
    gamdata_nr = get_gamdata_reduced_nr(gampart, param_nr, controls$gamdata),
    gamdata_combined = TRUE
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
  ncolNum <- ncol(get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")())
  
  sp_and_S <- get_gamdata(gampart, param_nr, controls$gamdata, what="sp_and_S")
  P <- sp_and_S[[1]][[1]] * sp_and_S[[2]][[1]]
  
  if(length(nlev)==1){
    layer <- arraym_block(ncolNum, nlev, penalty = controls$sp_scale(data) * P, 
                          name = makelayername(term, param_nr))
    #}else if(length(nlev)==2){
  }else{
    stop("am terms with more than 1 factor currently not supported.")
  }
  
  data_trafo_red = function() do.call("cbind", lapply(data[extractvar(byt)], int_0based))
  predict_trafo_red = function(newdata) do.call("cbind", lapply(newdata[extractvar(byt)],int_0based))
  
  data_trafo = function() do.call("cbind", c(list(
    get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")()
  ), lapply(data[extractvar(byt)], int_0based)))
  predict_trafo = function(newdata) do.call("cbind", c(
    list(get_gamdata(gampart, param_nr, controls$gamdata, what="predict_trafo")(newdata)),
    lapply(newdata[extractvar(byt)],int_0based)))
  
  list(
    data_trafo = data_trafo_red,
    predict_trafo = predict_trafo_red,
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
    input_dim = as.integer(length(nlev)),
    layer = layer,
    coef = function(weights) as.matrix(weights),
    gamdata_nr = get_gamdata_reduced_nr(gampart, param_nr, controls$gamdata),
    gamdata_combined = TRUE
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
  ncolNum <- ncol(get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")())
  
  # what type of interaction
  dimL = NULL
  simple <- suppressWarnings(extractval(term, "simple"))
  if(is.null(simple) || simple=="FALSE")
    dimL <- ncolNum
  
  sp_and_S <- get_gamdata(gampart, param_nr, controls$gamdata, what="sp_and_S")
  P <- sp_and_S[[1]][[1]] * sp_and_S[[2]][[1]]  
  
  layer <- vf_block(ncolNum, nlev[1], nlev[2], dim, dimL = dimL,
                    penalty = controls$sp_scale(data) * P, 
                    name = makelayername(term, param_nr), 
                    units = output_dim)
  
  rowSumsProd <- function(a,b) rowSums(a*b)
  
  data_trafo <- function() do.call("cbind", c(list(
    get_gamdata(gampart, param_nr, controls$gamdata, what="data_trafo")()
  ), lapply(data[extractvar(byt)], int_0based)))
  predict_trafo <- function(newdata) do.call("cbind", c(
    list(get_gamdata(gampart, param_nr, controls$gamdata, what="predict_trafo")(newdata)),
    lapply(newdata[extractvar(byt)],int_0based)))
  
  data_trafo_red <- function() do.call("cbind", lapply(data[extractvar(byt)], int_0based))
  predict_trafo_red <- function(newdata) do.call("cbind", lapply(newdata[extractvar(byt)],int_0based))
  
  list(
    data_trafo = data_trafo_red,
    predict_trafo = predict_trafo_red,
    input_dim = as.integer(length(nlev)),
    layer = layer,
    coef = function(weights) weights,
    partial_effect = function(weights, newdata=NULL){
      if(!is.null(newdata)) X <- predict_trafo(newdata) else
        X <- data_trafo()
      # +2 because 0-based and embd learns one additional empty value effect
      latent1 <- weights[[1]][X[,ncol(X)-1]+1,] 
      latent2 <- weights[[2]][X[,ncol(X)]+1,]
      latentprod <- do.call("cbind", lapply(1:ncolNum, function(i) 
        rowSumsProd(latent1[,(i-1)*dim + 1:dim],
                    latent2[,(i-1)*dim + 1:dim])))
      return(rowSumsProd(X[,1:(ncol(X)-2)], latentprod))
    },
    gamdata_nr = get_gamdata_reduced_nr(gampart, param_nr, controls$gamdata),
    gamdata_combined = TRUE
  )
  
}


#' Processor for additive factorization machine
#' 
#' @export
#' 
afm_processor <- function(term, data, output_dim, param_nr, controls){
  
  vars <- extractvar(term)
  nfac <- extractval(term, "fac")
  koption <- extractval(term, "k")
  # if(is.null(koption)) koption <- 10
  bsoption <- extractval(term, "bs")
  # if(is.null(bsoption)) bsoption <- "'tp'"
  # extract gam part
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  gamterms <- lapply(vars, function(var){
    
    gampart <- paste0("s(", var)
    
    if(!is.null(koption)){
      ko <- koption
      luv <- length(unique(data[[var]]))
      if(ko > luv/2) ko <- round(luv/2)
      gampart <- paste0(gampart, ", k = ", ko)
    }
    if(!is.null(bsoption)){
      gampart <- paste0(gampart, ", bs = ", bsoption)
    }
    
    gampart <- paste0(gampart, ")")

    evaluated_gam_term <- handle_gam_term(object = gampart, 
                                          data = data, 
                                          controls = controls)
    return(evaluated_gam_term)
    
  })
  
  input_dims <- sapply(gamterms, function(x) ncol(x[[1]]$X))
  penalties <- lapply(gamterms, function(x){
    
    sp_and_S <- create_penalty(x, controls$df, controls)[[1]]
    
    sp_and_S <- list(sp = 1, 
                     S = list(do.call("+", lapply(1:length(sp_and_S[[2]]), function(i)
                       sp_and_S[[1]][i] * sp_and_S[[2]][[i]]))))
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
    
    xsplit <- tf_split_multiple(x, input_dims)
    
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
      pairs <- create_pairs(length(gamterms), 2)
      if(is.null(newdata)) newdata <- list(data[vars])[rep(1,nrow(pairs))]
      lapply(1:nrow(pairs), function(i){
        ind_i <- pairs[i,"i"]
        ind_j <- pairs[i,"j"]
        rowSums(sapply(1:nfac, function(fl)
          kronecker( 
            (predict_gam_handler(gamterms[[ind_j]], newdata = newdata[[i]]) %*% 
               (weights[[pairs[i,"j"]]][[fl]])),
            predict_gam_handler(gamterms[[ind_i]], newdata = newdata[[i]]) %*% 
              (weights[[pairs[i,"i"]]][[fl]]))))
      })
    },
    coef = function(weights) as.matrix(weights)
  )
}

#' Processor for additive higher-order factorization machine
#' 
#' @export
#' 
ahofm_processor <- function(term, data, output_dim, param_nr, controls){
  
  vars <- extractvar(term)
  nfac <- extractval(term, "fac")
  order <- extractval(term, "order")
  koption <- extractval(term, "k")
  # if(is.null(koption)) koption <- 10
  bsoption <- extractval(term, "bs")
  # if(is.null(bsoption)) bsoption <- "'tp'"
  
  output_dim <- as.integer(output_dim)
  # extract mgcv smooth object
  gamterms <- lapply(vars, function(var){
    
    gampart <- paste0("s(", var)
    
    if(!is.null(koption)){
      ko <- koption
      luv <- length(unique(data[[var]]))
      if(ko > luv/2) ko <- round(luv/2)
      gampart <- paste0(gampart, "k = ", ko)
    }
    if(!is.null(bsoption)){
      gampart <- paste0(gampart, ", bs = ", bsoption)
    }
    
    gampart <- paste0(gampart, ")")
    
    evaluated_gam_term <- handle_gam_term(object = gampart, 
                                          data = data, 
                                          controls = controls)
    return(evaluated_gam_term)
    
  })
  
  input_dims <- sapply(gamterms, function(x) ncol(x[[1]]$X))
  penalties <- lapply(gamterms, function(x){
    
    sp_and_S <- create_penalty(x, controls$df, controls)[[1]]
    
    sp_and_S <- list(sp = 1, 
                     S = list(do.call("+", lapply(1:length(sp_and_S[[2]]), function(i)
                       sp_and_S[[1]][i] * sp_and_S[[2]][[i]]))))
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
    
    xsplit <- tf_split_multiple(x, input_dims)
    
    splineparts <- lapply(1:nfac, function(j) lapply(1:length(vars), function(i)
      splineparts[[j]][[i]](xsplit[[i]])))
    
    
    Dt <- lapply(1:nfac, function(j) 
      lapply(1:order, function(t) layer_add_identity(lapply(splineparts[[j]], 
                                                            function(c) tf$math$pow(c,t)))))
    
    res <- tf$multiply(1/order, layer_add_identity(lapply(1:nfac, function(j){
      
      Acurrent <- tf$math$add(tf$multiply((-1)^(order+1), Dt[[j]][[order]]),
                              tf$multiply((-1)^(order), tf$multiply(Dt[[j]][[1]],
                                                                    Dt[[j]][[order-1]])))
      
      if(order > 2){
        for(t in (order-2):1){
          
          Acurrent <- tf$math$add(Acurrent, tf$multiply((-1)^(t+1), tf$multiply(Acurrent,
                                                                                Dt[[j]][[t]])))
          
        }
      }
      
      return(Acurrent)
    
    })))

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
      pairs <- create_pairs(length(gamterms), order)
      if(is.null(newdata)){
        newdata <- data[vars]
        return(
          lapply(1:nrow(pairs), function(i){
            inds <- pairs[i,]
            rowSums(sapply(1:nfac, function(fl)
              Reduce("*", lapply(inds, function(ijk)
                predict_gam_handler(gamterms[[ijk]], newdata = newdata) %*% 
                  (weights[[ijk]][[fl]])
              ))
            ))
          })
        )
      }else{
        return(
          lapply(1:nrow(pairs), function(i){
            inds <- pairs[i,]
            rowSums(sapply(1:nfac, function(fl)
              Reduce("kronecker", lapply(inds, function(ijk)
                predict_gam_handler(gamterms[[ijk]], newdata = newdata) %*% 
                  (weights[[ijk]][[fl]])
              ))
            ))
          })
        )
      }
    },
    coef = function(weights) as.matrix(weights)
  )
}


#' Processor for higher-order factorization machine
#' 
#' @export
#' 
hofm_processor <- function(term, data, output_dim, param_nr, controls){
  
  vars <- extractvar(term)
  opts <- extractvals(term, c("fac", "la", "order"))
  nfac <- opts$fac
  la <- opts$la
  if(is.null(la)) la <- 0.1
  order <- opts$order
  if(is.null(opts$order)) order <- 2
  
  output_dim <- as.integer(output_dim)
  input_dims <- length(vars)
  
  weights <- lapply(1:nfac, function(j){
    
    function(object) layer_dense(object, name = paste0(makelayername(term, param_nr),
                                                       "_fac_", j), 
                                 units = output_dim, use_bias = FALSE, 
                                 kernel_regularizer = regularizer_l2(l = la)#, 
                                 #kernel_initializer = "zeros"
                                 )
    
  })
  
  layer = function(x, ...){
    
    lms <- lapply(1:nfac, function(j) weights[[j]](x))
    
    Dt <- lapply(1:nfac, function(j) 
      lapply(1:order, function(t) tf$math$pow(lms[[j]], t)))
    
    res <- tf$multiply(1/order, layer_add_identity(lapply(1:nfac, function(j){
      
      Acurrent <- tf$math$add(tf$multiply((-1)^(order+1), Dt[[j]][[order]]),
                              tf$multiply((-1)^(order), tf$multiply(Dt[[j]][[1]],
                                                                    Dt[[j]][[order-1]])))
      
      if(order > 2){
        for(t in (order-2):1){
          
          Acurrent <- tf$math$add(Acurrent, tf$multiply((-1)^(t+1), tf$multiply(Acurrent,
                                                                                Dt[[j]][[t]])))
          
        }
      }
      
      return(Acurrent)
      
    })))
    
    return(res)
    
  }
  
  data_trafo <- function() data[vars]
  predict_trafo <- function(newdata) newdata[vars]
  
  list(
    data_trafo = data_trafo,
    predict_trafo = predict_trafo,
    input_dim = as.integer(input_dims),
    layer = layer,
    coef = function(weights) as.matrix(weights)
  )
}


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

#' @export
layer_arram = function(a, b, pen=NULL, name="linearArrayRWT") {
  
  lal <- linearArrayRWT(units = c(a$shape[[2]], b$shape[[2]]), P = pen, name = name)
  return(lal(list(a,b)))
  
}

pen_vc <- function(P, strength, nlev)
{
  python_path <- system.file("python", package = "safareg")
  misc <- reticulate::import_from_path("misc", path = python_path)
  return(misc$squaredPenaltyVC(P = as.matrix(P), strength = strength, nlev = nlev))
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
  misc <- reticulate::import_from_path("misc", path = python_path)
  return(misc$LinearArrayRWT(units = units, P = as.matrix(P), name = name))
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

vf_block <- function(ncolNum, levFac1, levFac2, dim, dimL = NULL,
                     penalty = NULL, name = NULL, units = 1){
  
  fz <- factorization(
    dim,
    levFac1,
    levFac2,
    NULL,
    name_prefix = name,
    dimL = dimL)
  
  if(!is.null(penalty))
    penalty = pen_simple(penalty, 1)
  
  ret_fun <- function(x){
    
    a = tf_stride_cols(x, 1, ncolNum)
    b_c = tf$cast(tf_stride_cols(x,as.integer((ncolNum+1)),as.integer((ncolNum+2))), dtype="int32")
    fz_res = fz(b_c)
    
    basis_times_latent = tf$multiply(a, fz_res)
    if(is.null(dimL)){
      ret <- basis_times_latent %>% layer_dense(units = units, activation = "linear", 
                                                name = name, use_bias = FALSE, 
                                                kernel_regularizer = penalty)
    }else{
      ret <- tf$reduce_sum(basis_times_latent, axis = 1L)
    }
    
    return(ret)
    
    
  }
  
  return(ret_fun)
  
}

vvc_block <- function(ncolNum, levFac1, levFac2, penalty = NULL, name = NULL, units = 1, with_layer = TRUE){
  
  if(!is.null(penalty))
    penalty = pen_vc(penalty, strength=1, nlev=levFac1*levFac2)
  
  ret_fun <- function(x){
    
    b <- tf$one_hot(tf$cast(tf_stride_cols(x,as.integer((ncolNum+1))), dtype="int32"), 
                    depth=levFac1)
    c <- tf$one_hot(tf$cast(tf_stride_cols(x,as.integer((ncolNum+2))), dtype="int32"), 
                    depth=levFac2)
    
    if(length(b$shape)==2)
      b <- tf$squeeze(b, axis=1L)
    if(length(c$shape)==2)
      c <- tf$squeeze(c, axis=1L)
    
    ttp_layer(x[,as.integer(1:ncolNum)], b, c, 
              pen=penalty, name=name, units=units, with_layer=with_layer)
    
  }
  
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

factorization <- function(embedding_dim, xlev, ylev, zlev, name_prefix, dimL=NULL) {
  
  dotprod_org <- function(a,b) tf$math$reduce_sum(tf$multiply(a,b), axis=2L)
  
  if(!is.null(dimL)){
    
    # reshape_dimL <- function(x) tf$reshape(x, c(as.integer(dimL), 
    #                                             as.integer(embedding_dim)))
    
    dotprod <- function(a,b){ 
      
      a <- tf$split(a, num_or_size_splits = dimL, axis=2L)
      b <- tf$split(b, num_or_size_splits = dimL, axis=2L)
      reduced_latent <- lapply(1:length(a), function(i) dotprod_org(a[[i]],b[[i]]))
      return(tf$concat(reduced_latent, axis = 1L))
      
    }
    
    embedding_dim <- embedding_dim * dimL
    
  }else{
    dotprod <- dotprod_org
  }
  
  dot_fun <- keras_model_custom(name = "factorization", function(self) {
    self$x_embedding <-
      layer_embedding(
        input_dim = xlev,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = paste0(name_prefix, "_first_level_embedding")
      )
    self$y_embedding <-
      layer_embedding(
        input_dim = ylev,
        output_dim = embedding_dim,
        embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
        name = paste0(name_prefix, "_second_level_embedding")
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
          input_dim = zlev,
          output_dim = embedding_dim,
          embeddings_initializer = initializer_random_uniform(minval = 0, maxval = 0.05),
          name = paste0(name_prefix, "_third_level_embedding")
        )
      self$dot <-
        layer_lambda(
          f = function(x) {
            dotprod(tf$multiply(x[[1]], x[[2]]), x[[3]])
          }
  )
  return(function(x, mask = NULL, training = FALSE) {
    y <- tf_stride_cols(x, 2L)
    z <- tf_stride_cols(x, 3L)
    x <- tf_stride_cols(x, 1L)
    x_embedding <- self$x_embedding(x)
    y_embedding <- self$y_embedding(y)
    z_embedding <- self$z_embedding(z)
    self$dot(list(x_embedding, 
                  y_embedding, 
                  z_embedding))
  })
    }
  })

return(dot_fun)

}

int_0based <- function(x) as.integer(x)-1L


