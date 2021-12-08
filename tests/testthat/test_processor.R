context("Processors")

test_that("vc_processor", {
  
  n <- 40
  data = data.frame(a=rnorm(n), b=rnorm(n), c=gl(n/2,2), d=gl(n/2,2))
  term="vc(te(a,b), by=c(c,d))"
  controls <- penalty_control()
  controls$gamdata <- deepregression::precalc_gam(list(as.formula(paste0("~ ", term))), data, controls)
  expect_equal(vc_processor(term, data, 1, 1, controls)$input_dim, 2)
  term="vc(te(a,b), by=c)"
  controls$gamdata <- precalc_gam(list(as.formula(paste0("~ ", term))), data, controls)
  expect_equal(vc_processor(term, data, 1, 1, controls)$input_dim, 1)
  term="vc(s(a, zerocons = FALSE), by=c(c,d))"
  controls$gamdata <- precalc_gam(list(as.formula(paste0("~ ", term))), data, controls)
  expect_equal(vc_processor(term, data, 1, 1, controls)$input_dim, 2)
  term="vc(s(a, zerocons = FALSE), by=c)" 
  controls$gamdata <- precalc_gam(list(as.formula(paste0("~ ", term))), data, controls)
  expect_equal(vc_processor(term, data, 1, 1, controls)$input_dim, 1)
  
})

test_that("afm_processor", {
  
  functions_bi = list(
    function(x, z, sx = 0.3, sz = 0.4) {
      (pi^sx * sz) * (1.2 * exp(-(x - 0.2)^2/sx^2 - (z - 0.3)^2/sz^2) + 
                        0.8 * exp(-(x - 0.7)^2/sx^2 - (z - 0.8)^2/sz^2))
    },
    function(x, z) {
      cos(x)*sin(z)
    },
    function(x, z) {
      (x/2)*(z/2)
    }
  )
  
  n <- 1000
  p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  X <- as.data.frame(X)
  colnames(X) <- paste0("x", 1:3)
  
  pred1 <- functions_bi[[1]](X[,1], X[,2]) 
  pred2 <- functions_bi[[2]](X[,1], X[,3])
  pred3 <- functions_bi[[3]](X[,2], X[,3])
  
  preds <- list(pred1, pred2, pred3)
  
  y <- rnorm(n = n, pred1 + pred2 + pred3, 0.1)
  data = cbind(X, y = y)
  
  mod <- deepregression(y = y,
                        list_of_formulas = list(
                          ~ afm(x1, x2, x3, fac = 10), 
                          ~1
                        ),
                        data = data, 
                        optimizer = tf$keras$optimizers$Adam(learning_rate = 1e-3),
                        additional_processors = list(afm = afm_processor)
  )
  
  expect_is(mod, "deepregression")
  
  hist <- mod %>% fit(epochs = 2, batch_size = 128L, 
                      early_stopping = TRUE, patience = 50L,
                      verbose = FALSE)
  
  pred <- mod %>% fitted()
  
  expect_equal(dim(pred), c(1000,1))
  
  w <- mod$model %>% get_weights()
  expect_length(w, 32)
  
  data_pe <- mod$init_params$parsed_formulas_contents[[1]][[1]]$data_trafo()
  expect_equal(dim(data_pe), c(1000,30))
  
})