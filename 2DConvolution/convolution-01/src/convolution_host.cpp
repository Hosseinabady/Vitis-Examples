#include "convolution_host.h"

#include <sys/time.h>

double getTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}


double hardware_start;
double hardware_end;
double hardware_execution_time;


void convolution_sw(
		std::vector<data_type,aligned_allocator<data_type>> input,
		std::vector<data_type,aligned_allocator<data_type>> output,
		std::vector<data_type,aligned_allocator<data_type>> mask,
		int        n,
		int        m,

		int        p,
		int        q
)
{

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			data_type out_data = 0;
			for (int h = 0; h < p; h++) {
				for (int w = 0; w < q; w++) {
					if ( (i-p/2+h >= 0 && i-p/2+h < n) &&
						 (j-q/2+w >= 0 && j-q/2+w < m)
						)
						out_data += input[(i-p/2+h)*m+j-q/2+w]*mask[h*q+w];
				}
			}
			output[i*m+j] = out_data;
		}
	}

}

int main(int argc, char** argv)
{

	std::cout << " Hello 2D Convolution " << std::endl;

	int n = 256;
	int m = 256;

	int p = 5;
	int q = 5;




	cl_int err;




	std::vector<data_type,aligned_allocator<data_type>> input_h(sizeof(data_type) * n * m);
	std::vector<data_type,aligned_allocator<data_type>> output_h_hw(sizeof(data_type) * n*m);
	std::vector<data_type,aligned_allocator<data_type>> output_h_sw(sizeof(data_type) * n*m);
	std::vector<data_type,aligned_allocator<data_type>> mask_h(sizeof(data_type) * p*q);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			input_h[i*m+j] = rand()/(1.0*RAND_MAX);
			output_h_hw[i*m+j] = rand()/(1.0*RAND_MAX);
			output_h_sw[i*m+j] = rand()/(1.0*RAND_MAX);
		}
	}



	for (int i = 0; i < p; i++) {
		for (int j = 0; j < q; j++) {
			mask_h[i*q+j] = rand()/(1.0*RAND_MAX);
		}
	}


	convolution_sw(input_h, output_h_sw, mask_h, n, m, p, q);

    std::vector<cl::Device> devices = get_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Found Device=" << device_name.c_str() << std::endl;

    //Creating Context and Command Queue for selected device
    cl::Context context(device);
    cl::CommandQueue cq(context, device);

    // Import XCLBIN
    xclbin_file_name = argv[1];
    cl::Program::Binaries convolution_accel_bins = import_binary_file();

    // Program and Kernel
    devices.resize(1);
    cl::Program program(context, devices, convolution_accel_bins);
    cl::Kernel krnl_convolution_accel_(program, "convolution_accel");




    OCL_CHECK(err, cl::Buffer buffer_input(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    		sizeof(data_type) * n * m, input_h.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_mask(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    		sizeof(data_type) * p * q, mask_h.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_output   (context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
    		sizeof(data_type) * n, output_h_hw.data(), &err));



    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(0, buffer_input));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(1, buffer_output));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(2, buffer_mask));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(3, n));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(4, m));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(5, p));
    OCL_CHECK(err, err = krnl_convolution_accel_.setArg(6, q));

    hardware_start = getTimestamp();

    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_input, buffer_mask},0/* 0 means from host*/));

    OCL_CHECK(err, err = cq.enqueueTask(krnl_convolution_accel_));


    OCL_CHECK(err, err = cq.enqueueMigrateMemObjects({buffer_output},CL_MIGRATE_MEM_OBJECT_HOST));
    cq.finish();
    hardware_end = getTimestamp();

    hardware_execution_time = (hardware_end-hardware_start)/(1000);
    std::cout << "krnl_convolution_accel_ hardware execution time  " << hardware_execution_time << " ms elapsed" << std::endl;

    int status = 0;
    for(u32 i=0; i< n; i++) {
    	for(u32 j=0; j< n*m; j++) {
    		data_type diff = fabs(output_h_sw[i*m+j]-output_h_hw[i*m+j]);
    		if(diff > 0.1 || diff != diff){
    			std::cout << "error occurs at (" << i << ", " << j << ") with value output_h_hw = " <<  output_h_hw[i*m+j] << ", should be output_h_sw = " << output_h_sw[i*m+j] << std::endl;
    			status = -1;
    			break;
    		}
    	}
	}

	if(!status) {
		std::cout << "Validation PASSED!"<< std::endl;
	} else {
		std::cout << "Validation FAILED!" << std::endl;
	}


	printf("\rBye 2D Convolution!\n\r");

	return status;

}


