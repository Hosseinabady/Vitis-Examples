#include "sgemv_host.hpp"

#include <sys/time.h>

double getTimestamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + tv.tv_sec*1e6;
}


double hardware_start;
double hardware_end;
double hardware_execution_time;

int main(int argc, char** argv)
{

	std::cout << " Hello SGEMV " << std::endl;

	u32 n = ROW_SIZE_MAX;
	u32 m = COL_SIZE_MAX;

	DATA_TYPE alpha = 2.34;
	DATA_TYPE beta  = 1.26;

	cl_int err;

	std::vector<DATA_TYPE,aligned_allocator<DATA_TYPE>> A_h(sizeof(DATA_TYPE) * n * m);
	std::vector<DATA_TYPE,aligned_allocator<DATA_TYPE>> y_h_hw(sizeof(DATA_TYPE) * n);
	std::vector<DATA_TYPE,aligned_allocator<DATA_TYPE>> y_h_sw(sizeof(DATA_TYPE) * n);
	std::vector<DATA_TYPE,aligned_allocator<DATA_TYPE>> x_h(sizeof(DATA_TYPE) * m);

	for(size_t i = 0 ; i < n ; i++){
		for (size_t j = 0; j < m; j++) {
			A_h[i*m+j] = rand()/(1.0*RAND_MAX);
		}
	 }

	for(size_t i = 0 ; i < n ; i++){
		y_h_hw[i] = y_h_sw[i] = rand()/(1.0*RAND_MAX);
	}

	for(size_t i = 0 ; i < m ; i++){
		x_h[i] = rand()/(1.0*RAND_MAX);
	}


	for (size_t i = 0; i < n; i++) {
		DATA_TYPE sum = beta*y_h_sw[i];
		for (size_t j = 0; j < m; j++) {
			sum += alpha*A_h[i*m+j]*x_h[j];
		}
		y_h_sw[i] = sum;
	}

    std::vector<cl::Device> devices = get_devices();
    cl::Device device = devices[0];
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Device=" << device_name.c_str() << std::endl;


    cl::Context context(device);
    cl::CommandQueue q(context, device);


    xclbin_file_name = argv[1];
    cl::Program::Binaries sgemv_bins = import_binary_file();


    devices.resize(1);
    cl::Program program(context, devices, sgemv_bins);
    cl::Kernel krnl_sgemv(program, "sgemv");




    OCL_CHECK(err, cl::Buffer buffer_A(context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
    		sizeof(DATA_TYPE) * n * m, A_h.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_y   (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
    		sizeof(DATA_TYPE) * n, y_h_hw.data(), &err));

    OCL_CHECK(err, cl::Buffer buffer_x(context,CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
    		sizeof(DATA_TYPE) * m, x_h.data(), &err));


    OCL_CHECK(err, err = krnl_sgemv.setArg(0, buffer_A));
    OCL_CHECK(err, err = krnl_sgemv.setArg(1, buffer_x));
    OCL_CHECK(err, err = krnl_sgemv.setArg(2, buffer_y));
    OCL_CHECK(err, err = krnl_sgemv.setArg(3, n));
    OCL_CHECK(err, err = krnl_sgemv.setArg(4, m));
    OCL_CHECK(err, err = krnl_sgemv.setArg(5, alpha));
    OCL_CHECK(err, err = krnl_sgemv.setArg(6, beta));

    hardware_start = getTimestamp();

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A, buffer_y, buffer_x},0));

    OCL_CHECK(err, err = q.enqueueTask(krnl_sgemv));

    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_y},CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    hardware_end = getTimestamp();

    hardware_execution_time = (hardware_end-hardware_start)/(1000);
    std::cout << "sgemv hardware execution time  " << hardware_execution_time << " ms elapsed" << std::endl;

    int status = 0;
	for(u32 i=0; i< n; i++) {
		DATA_TYPE diff = fabs(y_h_sw[i]-y_h_hw[i]);
		if(diff > 0.1 || diff != diff){
			std::cout << "error occurs at " << i << " with value y_h_hw = " <<  y_h_hw[i] << ", should be y_h_sw = " << y_h_sw[i] << std::endl;
			status = -1;
			break;
	    }


	}
	if(!status) {
		std::cout << "Validation PASSED!"<< std::endl;
	} else {
		std::cout << "Validation FAILED!" << std::endl;
	}


	printf("\rBye SGEMV!\n\r");

	return status;

}


