function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
  visible_data = sample_bernoulli(visible_data); %Added in based on Q8

  p_h_given_v_t0 = visible_state_to_hidden_probabilities(rbm_w, visible_data);
  h_tilde_t0 = sample_bernoulli(p_h_given_v_t0);
  expected_vi_hj_t0 = configuration_goodness_gradient(visible_data, h_tilde_t0);

  p_v_given_h = hidden_state_to_visible_probabilities(rbm_w, h_tilde_t0);
  v_tilde_t1 = sample_bernoulli(p_v_given_h);

  p_h_given_v_t1 = visible_state_to_hidden_probabilities(rbm_w, v_tilde_t1);
  h_tilde_t1 = sample_bernoulli(p_h_given_v_t1);
  expected_vi_hj_t1 = configuration_goodness_gradient(v_tilde_t1, h_tilde_t1);

  %This line below was added in based on the instructions in question 7
  expected_vi_hj_t1 = configuration_goodness_gradient(v_tilde_t1, p_h_given_v_t1);

  ret = expected_vi_hj_t0 - expected_vi_hj_t1;
end
