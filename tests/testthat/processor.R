context("Processors")

test_that("vc_processor", {
  
  n <- 40
  data = data.frame(a=rnorm(n), b=rnorm(n), c=gl(n/2,2), d=gl(n/2,2))
  term="vc(te(a,b), by=c(c,d))"
  expect_equal(vc_processor(term, data, 1, 1, penalty_control())$input_dim, 27)
  term="vc(te(a,b), by=c)"
  expect_equal(vc_processor(term, data, 1, 1, penalty_control())$input_dim, 26)
  term="vc(s(a), by=c(c,d))"
  expect_equal(vc_processor(term, data, 1, 1, penalty_control())$input_dim, 12)
  term="vc(s(a), by=c)" 
  expect_equal(vc_processor(term, data, 1, 1, penalty_control())$input_dim, 11)
  
})