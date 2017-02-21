package shenpeng;
import static jcuda.driver.JCudaDriver.*;
import java.io.*;
import jcuda.*;
import jcuda.driver.*;

	
	 class JCudaAdd implements Serializable
	{
	   
        public  float getPi(int d)
		    {
	   // 初始化设备；创建上下文
	        cuInit(0);
	        CUdevice device = new CUdevice();
	        cuDeviceGet(device, 0);
	        CUcontext context = new CUcontext();
	        cuCtxCreate(context, 0, device);

	        // Load the ptx file.
	        CUmodule module = new CUmodule();
	        cuModuleLoad(module,"JCudaVectorAddKernel.ptx" );

	        // Obtain a function pointer to the "add" function.
	        CUfunction function = new CUfunction();
	        cuModuleGetFunction(function, module, "add");

	        int numElements = 100000;
	        float m=(float)1.0/numElements;
	        int h=numElements/4;

	        // Allocate and fill the host input data
	        float hostInputA[] = new float[h];
	        float hostInputB[] = new float[h];
	        for(int i = 0; i < h; i++)
	        {   
	            hostInputA[i] = (float)(4.0*i+d-0.5);
	            hostInputB[i] = (float)m;
	        }

	        // Allocate the device input data, and copy the
	        // host input data to the device
	        CUdeviceptr deviceInputA = new CUdeviceptr();
	        cuMemAlloc(deviceInputA, h * Sizeof.FLOAT );
	        cuMemcpyHtoD(deviceInputA, Pointer.to(hostInputA),
	        		h * Sizeof.FLOAT );
	        CUdeviceptr deviceInputB = new CUdeviceptr();
	        cuMemAlloc(deviceInputB, h * Sizeof.FLOAT );
	        cuMemcpyHtoD(deviceInputB, Pointer.to(hostInputB),
	        		h * Sizeof.FLOAT );

	        // Allocate device output memory
	        CUdeviceptr deviceOutput = new CUdeviceptr();
	        cuMemAlloc(deviceOutput, h * Sizeof.FLOAT);

	        // Set up the kernel parameters: A pointer to an array
	        // of pointers which point to the actual values.
	        Pointer kernelParameters = Pointer.to(
	            Pointer.to(new int[]{h}),
	            Pointer.to(deviceInputA),
	            Pointer.to(deviceInputB),
	            Pointer.to(deviceOutput)
	        );

	        // Call the kernel function.
	        int blockSizeX = 256;
	        int gridSizeX = (int)Math.ceil((double)h/ blockSizeX);
	        cuLaunchKernel(function,
	            gridSizeX,  1, 1,      // Grid dimension
	            blockSizeX, 1, 1,      // Block dimension
	            0, null,               // Shared memory size and stream
	            kernelParameters, null // Kernel- and extra parameters
	        );
	        cuCtxSynchronize();

	        // Allocate host output memory and copy the device output
	        // to the host.
	        float hostOutput[] = new float[h];
	        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput,
	        		h * Sizeof.FLOAT);

	       
	        //计算pi
	        float pi=0;
	        for(int i = 0; i < h; i++){
	          pi+=hostOutput[i];	
	        }
	        
	       
	        
            // Clean up.
	        cuMemFree(deviceInputA);
	        cuMemFree(deviceInputB);
	        cuMemFree(deviceOutput);
	        
	        return pi;
	      }
		
		 }
		 
	     
		 


