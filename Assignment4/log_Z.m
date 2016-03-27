function lz = log_Z(rbm_w)
  %The assumption is that there are far fewer hidden states
  %We start by generating a matrix where the columns are all the
  %possible states
  num_hidden = size(rbm_w, 1);
  all_hidden_configs = (dec2bin( 0:(2^num_hidden - 1), num_hidden) - '0')';

  %Next, we compute the h_prime matrix
  %This size of h_prime is (num_visible, 2^num_hidden)
  h_prime = rbm_w'*all_hidden_configs;
  col_products = prod(1 + exp(h_prime));

  lz = log(sum(col_products));
  