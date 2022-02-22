create_pairs <- function(nfeat, D){
  
  if(D==2){
    df <- data.frame(i = NA, j = NA)
    for(i in 1:(nfeat-1)){
      for(j in (i+1):nfeat){
        df <- rbind(df, data.frame(i = i, j = j))
      }
    }
  }else if(D==3){
    df <- data.frame(i = NA, j = NA, k = NA)
    for(i in 1:(nfeat-2)){
      for(j in (i+1):(nfeat-1)){
        for(k in (j+1):nfeat){
          df <- rbind(df, data.frame(i = i, j = j, k = k))
        }
      }
    }
  }else{
    stop("Not implemented.")
  }
  return(df[-1,])
  
}