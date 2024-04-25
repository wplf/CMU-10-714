#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void println(const float* array, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            std::cout << array[i * n + j] << " ";
        }
        std::cout << "\n";
    }
}
float* matrix_mul(const float* in1, const float* in2, int m, int n, int k){
    float* out = (float*) malloc(sizeof(float) * m * k);

    for(int i = 0; i < m; i++){
        for(int j = 0; j < k; j++){
            out[i * k + j] = 0;
            for(int l = 0; l < n; l++){
                out[i * k + j] += in1[i * n + l] * in2[l * k + j];
            }
        }
    }
    return out;
}

float* softmax(float* X, int m, int n){
    // return m, n
    float* out = (float*)malloc(sizeof(float) * m * n);
    for(int i = 0; i < m; i++){
        float partial = 0; 
        for(int j = 0; j < n; j++){
            out[i * n + j] = exp(X[i * n + j]);
            partial += out[i * n + j];
        }
        
        for(int j = 0; j < n; j++){
            out[i * n + j] = out[i * n + j]  / partial;
        }
        // std::cout << "partial : " << partial << '\n';
    }
    // println(out, m, n);

    return out;
}

float * transpose(const float* mat, int m, int n){
    float* out = (float *) malloc(sizeof(float) * n * m);
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            out[i * m + j] = mat[j * n + i]; 
    return out;
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, int m, int n, int k,
								  float lr, int batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (int): number of examples
     *     n (int): input dimension
     *     k (int): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // std::cout << "X: " << X[0] <<  " y: " <<  int(y[0]) << std::endl;

    for(int i = 0; i < m; i += batch){
        const float *X_batch = &X[i * n];
        const unsigned char *y_batch = &y[i];
        float* z1 = matrix_mul(X_batch, theta, batch, n, k);

        // std::cout << "###X_batch\n";
        // println(X_batch, batch, n);
        // std::cout << "###theta\n";
        // println(theta, n, k);
        // std::cout << "###z1\n";
        // println(z1, batch, k);

        float *softmax_out = softmax(z1, batch, k); // batch, k
        // std::cout << "###softmax_out\n";
        // println(softmax_out, batch, k);

        int *I_y = (int *)malloc(sizeof(int) * batch * k); // batch, k
        for(int s = 0; s < batch; s++){
            for(int t = 0; t < k; t++){
                if(t == y_batch[s]) I_y[s * k + t] = 1;
                else I_y[s * k + t] = 0;
            }
        }

        float *out_grad = (float *)malloc(sizeof(float) * batch * k); // batch, k

        for(int s = 0; s < batch; s++){
            for(int t = 0; t < k; t++){
                out_grad[s * k + t] = softmax_out[s * k + t] - I_y[s * k + t];
            }
        }
        // n, batch @ batch * k
        float *theta_grad =  matrix_mul(transpose(X_batch, batch, n), out_grad, n, batch, k) ;
        for(int s = 0; s < n; s++)
            for(int t = 0; t < k; t++)
                theta[s * k + t] = theta[s * k + t] - lr * theta_grad[s * k + t] / batch;
            
        
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
