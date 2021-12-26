#' Initialize a Factorized Structured Regression Model
#' 
#' @param ... see \code{deepregression}
#' @param additional_processor defines how processors of this package are named
#' @return a model of class \code{safareg} and \code{deepregression}
#' 
#' @details The names of the processors can be choosen freely, but
#' note that methods for the class \code{safareg} might depend on this
#' naming convention.
#' 
safareg <- function(...,
                    additional_processors = list(fac = fac_processor,
                                                 facz = fz_processor,
                                                 vc = am_processor,
                                                 vfacz = vf_processor))
{
  
  mod <- deepregression(...,
                        additional_processors = additional_processors)
  class(mod) <- c("safareg", class(mod))
  
}