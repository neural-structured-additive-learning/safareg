#' Extract layer weights / coefficients from model
#'
#' @param object a safareg model
#' @param which_param integer, indicating for which distribution parameter
#' coefficients should be returned (default is first parameter)
#' @param type either NULL (all types of coefficients are returned),
#' "linear" for linear coefficients or "smooth" for coefficients of 
#' smooth terms
#' @param ... not used
#'
#' @importFrom stats coef
#' @method coef safareg
#' @rdname methodsSAFAREG
#' @export
#'
coef.safareg <- function(
  object,
  which_param = 1,
  type = NULL,
  ...
)
{
 
  pfc <- object$init_params$parsed_formulas_contents[[which_param]]
  to_return <- get_type_pfc(pfc, type)
  
  names <- get_names_pfc(pfc)[as.logical(to_return)]
  pfc <- pfc[as.logical(to_return)]
  check_names <- names
  check_names[check_names=="(Intercept)"] <- "1"
  
  postfix_needed <- grepl("facz", check_names) 

  coefs_nopostfix <- lapply(which(!postfix_needed), function(i) 
    pfc[[i]]$coef(get_weight_by_name(object, check_names[i], which_param)))
  
  coefs_postfix <- mapply(c,
                          lapply(which(postfix_needed), function(i) list(
                            pfc[[i]]$coef(get_weight_by_name(object, check_names[i], which_param, 
                                                             postfixes = "_first_level_embedding")))),
                          lapply(which(postfix_needed), function(i) list(
                            pfc[[i]]$coef(get_weight_by_name(object, check_names[i], which_param, 
                                                             postfixes = "_second_level_embedding")))),
                          SIMPLIFY = FALSE
  )

  coefs <- c(coefs_nopostfix, coefs_postfix)[match(1:length(names), c(which(!postfix_needed), which(postfix_needed)))]
    
  names(coefs) <- names
  
  return(coefs)
  
}


#' Partial Predictions Method for safareg
#' 
#' @param object,names,return_matrix,which_param,newdata see \code{?get_partial_effect} 
#' @param latentdim number of latent dimension for AFM or AHOFM
#' 
#' @export
#' 
get_partial_effect_latent <- function(object, names=NULL, return_matrix = FALSE, 
                                      which_param = 1, newdata = NULL, latentdim, ...)
{
  
  postfixes_fun <- function(varlen) paste0("te_", rep(1:varlen, each=latentdim), 
                                           "_fac_", rep(1:latentdim, varlen))
  
  names_pfc <- get_names_mod(object, which_param)
  names <- if(!is.null(names)) intersect(names, names_pfc) else names_pfc
  
  if(length(names)==0)
    stop("Cannot find specified name(s) in additive predictor #", which_param,".")
  
  res <- lapply(names, function(name){
    w <- which(name==names_pfc)
    
    if(name=="(Intercept)") name <- "1"
    if(grepl("afm\\(", name) | grepl("ahofm\\(", name)){
      
      varlen <- length(extractvar(name))
      weights <- get_weight_by_name(object, name = name, param_nr = which_param, 
                                    postfixes = postfixes_fun(varlen))
      weights <- split(weights, rep(1:varlen, each=latentdim))
      
    }else{
      
      weights <- get_weight_by_name(object, name = name, param_nr = which_param, ...)
      
    }
    
    pe_fun <- object$init_params$parsed_formulas_contents[[which_param]][[w]]$partial_effect
    if(is.null(pe_fun)){
      #warning("Specified term does not have a partial effect function. Returning weights.")
      return(weights)
    }else{
      return(pe_fun(weights, newdata))
    }
  })
  if(length(res)==1) return(res[[1]]) else return(res)
  
  
  
}