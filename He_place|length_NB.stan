
data {
  int<lower=1> N;  // total number of observations
  int place[N];  // response variable
  int unit_length[N];
}

parameters {
  real<lower = 1> lambda;
  real<lower = 0> mu; 
  real<lower = 0> phi; // 
}

transformed parameters {
  real lprior = 0;  // prior contributions to the log posterior
  lprior += gamma_lpdf(lambda | 3, 3);
  lprior += gamma_lpdf(mu | 1, 1);
  lprior += gamma_lpdf(phi | 1, 1);
}

model {
  target += lprior; // prior
  
  for (i in 1:N) {
    unit_length[i] ~ neg_binomial_2(mu, phi) T[1, ]; // truncated
    place[i] ~ poisson(lambda) T[1, unit_length[i]];
  }
}

//generated quantities {
  //int length_rep[N];
  //int place_rep[N];
  
  //for (i in 1:N) {
    
    //length_rep[i] = neg_binomial_2_rng(mu, phi);
    //place_rep[i] = poisson_rng(lambda);
  //}
//}
