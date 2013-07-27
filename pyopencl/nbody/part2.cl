__kernel void part2(__global float4* pos, __global float4* vel, __local float4* postemp, float dt)
{
    // get our index in the array
    unsigned int i = get_global_id(0);
    unsigned int li = get_local_id(0);

    unsigned int n = get_global_size(0);
    unsigned int ls = get_local_size(0);
    unsigned int nb = n/ls;
    float4 a = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    float4 p = pos[i];
    float4 v = vel[i];

    for(int k=0;k<nb;k++){
        postemp[li]= pos[k*ls+li];

        barrier(CLK_LOCAL_MEM_FENCE);
        for(int j=0;j<ls;j++){
            float4 ptemp = postemp[j];
            float4 diff = ptemp - p;
            float dist = length(diff);
            float dist3 = dist*dist*dist;

            a += 10.0f*diff/(dist3+000.1);
            a -= 15.0f*diff/(dist3*dist3*dist3+000.1);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    p += dt*v + dt*dt*a;
    v += dt*a - 0.1*v*dt;

    //update the arrays with our newly computed values
    pos[i] = p;
    vel[i] = v;
}
