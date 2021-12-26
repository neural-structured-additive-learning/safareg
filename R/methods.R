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
