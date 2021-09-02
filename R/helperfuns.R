# TensorFlow repeat function which is not available for TF 2.0
tf_repeat <- function(a, dim)
  tf$reshape(tf$tile(tf$expand_dims(a, axis = -1L),  c(1L, 1L, dim)), 
             shape = list(-1L, a$shape[[2]]*dim))

# Row-wise tensor product using TensorFlow
tf_row_tensor <- function(a,b)
{
  tf$multiply(
    tf_row_tensor_left_part(a,b),
    tf_row_tensor_right_part(a,b)
  )
}

tf_row_tensor_left_part <- function(a,b)
{
  tf_repeat(a, b$shape[[2]])
}

tf_row_tensor_right_part <- function(a,b)
{
  tf$tile(b, c(1L, a$shape[[2]]))
}
