# GeoAccM

Matlab version of geodesic acceleration.

Matlab used to directly code forwardprop, backprop, congugate gradient descent, etc.

Natural Gradient: To run with natural gradient, use Matlab and run nnet_demo_2.m with last call to nnet_train_ng() uncommented

Geodesic Acceleration: To run with geodesic acceleration added, use Matlab and run nnet_demo_2.m with last call to nnet_train_geo() uncommented.

GeoAcc currently vastly improves convergence rate for deep networks, especially on certain data sets like "curves".

