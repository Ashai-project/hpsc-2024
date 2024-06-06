#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>


const int nx = 41;
const int ny = 41;
const int nt = 500;
const int nit = 50;
const double dx = 2.0 / (nx - 1);
const double dy = 2.0 / (ny - 1);
const double dt = 0.01;
const double rho = 1.0;
const double nu = 0.02;

typedef std::vector<double> Matrix;

Matrix zeros(int size) {
    return Matrix(size, 0.0);
}

void write_csv(const std::string &filename, const Matrix &data, int rows, int cols) {
    std::ofstream file(filename);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i * cols + j];
            if (j != cols - 1) file << ",";
        }
        file << "\n";
    }
}

void CHECK_ERROR(cudaError_t err, const std::string& filename, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << '[' << cudaGetErrorString(err) << "] in " << filename << ", line " << line << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void build_up_b(double* b, double* u, double* v, int nx, int ny, double rho, double dt, double dx, double dy) {
    int j = blockIdx.x  ;
    int i =  threadIdx.x;
    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        // b[j * nx + i] = rho * (1 / dt * 
        //                       ((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2 * dx) + 
        //                        (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy)) - 
        //                       pow((u[j * nx + (i + 1)] - u[j * nx + (i - 1)]) / (2 * dx), 2) - 
        //                       2 * ((u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy) * 
        //                            (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2 * dx)) - 
        //                       pow((v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2 * dy), 2));
        b[j* nx +i] = rho * (1 / dt * 
                                ((u[j* nx +i+1] - u[j* nx +i-1]) / (2 * dx) + 
                                 (v[(j+1)* nx +i] - v[(j-1)* nx +i]) / (2 * dy)) - 
                                pow((u[j* nx +i+1] - u[j* nx +i-1]) / (2 * dx), 2) - 
                                2 * ((u[(j+1)* nx +i] - u[(j-1)* nx +i]) / (2 * dy) * 
                                     (v[j* nx +i+1] - v[j* nx +i-1]) / (2 * dx)) - 
                                pow((v[(j+1)* nx +i] - v[(j-1)* nx +i]) / (2 * dy), 2));
    }
}

__global__ void build_up_p(double* p, double* p1, double* pn, double* b, int nx, int ny, double dt, double dx, double dy) {
    int j = blockIdx.x ;
    int i =  threadIdx.x;
    if(j* nx +i>=nx*ny && j* nx +i<0)
        return;

    for (int it = 0; it < nit; it++){
        pn[j * nx + i] = p[j * nx + i];
        p[j * nx + i] = (dy*dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
                                        dx*dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
                                        b[j * nx + i] * dx*dx * dy*dy) / 
                                        (2 * (dx*dx + dy*dy));
        p1[j * nx + nx - 1] = p[j * nx + nx - 2];
        p[i] = p1[nx + i];
        p1[j * nx] = p[j * nx + 1];
        p[j * nx + i]=p1[j * nx + i];
        p[(ny - 1) * nx + i] = 0;
    }
}

__global__ void build_up_unvn(double* un, double* u, double* vn, double* v, int nx, int ny) {
    int j = blockIdx.x ;
    int i =  threadIdx.x;
    if(j* nx +i>=nx*ny && j* nx +i<0)
        return;
    un[j * nx + i] =u[j * nx + i];
    vn[j * nx + i]=v[j * nx + i];
}

__global__ void build_up_uv(double* u, double* un, double* v, double* vn,double* p, int nx, int ny,double rho, double dt, double dx, double dy,double nu) {
    int j = blockIdx.x ;
    int i =  threadIdx.x;
    if(j* nx +i>=nx*ny && j* nx +i<0)
        return;
    u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
                          un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
                          dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
                          nu * dt / (dx*dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
                          nu * dt / (dy*dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);

    v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
                          vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
                          dt / (2 * rho * dx) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
                          nu * dt / (dx*dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
                          nu * dt / (dy*dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
    u[i] = 0;
    u[j * nx] = 0;
    u[j * nx + nx - 1] = 0;
    u[(ny - 1) * nx + i] = 1;
    v[i] = 0;
    v[(ny - 1) * nx + i] = 0;
    v[j * nx] = 0;
    v[j * nx + nx - 1] = 0;
}


int main() {
    Matrix pn;
    Matrix u = zeros(nx * ny);
    Matrix v = zeros(nx * ny);
    Matrix p = zeros(nx * ny);
    Matrix b = zeros(nx * ny);

    double *d_u, *d_v, *d_p,*d_p1, *d_b, *d_pn,*d_un, *d_vn;

    CHECK_ERROR(cudaMalloc(&d_u, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_v, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_p, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_p1, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_pn, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_b, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_un, nx * ny * sizeof(double)), __FILE__, __LINE__);
    CHECK_ERROR(cudaMalloc(&d_vn, nx * ny * sizeof(double)), __FILE__, __LINE__);

    CHECK_ERROR(cudaMemcpy(d_u, &u[0], nx * ny * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    CHECK_ERROR(cudaMemcpy(d_v, &v[0], nx * ny * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    CHECK_ERROR(cudaMemcpy(d_p, &p[0], nx * ny * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    CHECK_ERROR(cudaMemcpy(d_b, &b[0], nx * ny * sizeof(double), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    
    for (int n = 0; n < nt; n++) {
        build_up_b<<<(nx*ny+nx-1)/nx, nx>>>(d_b, d_u, d_v, nx, ny, rho, dt, dx, dy);
        CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);
        // for (int j = 1; j < ny-1; j++) {
        //     for (int i = 1; i < nx-1; i++) {
        //         b[j* nx +i] = rho * (1 / dt * 
        //                         ((u[j* nx +i+1] - u[j* nx +i-1]) / (2 * dx) + 
        //                          (v[(j+1)* nx +i] - v[(j-1)* nx +i]) / (2 * dy)) - 
        //                         pow((u[j* nx +i+1] - u[j* nx +i-1]) / (2 * dx), 2) - 
        //                         2 * ((u[(j+1)* nx +i] - u[(j-1)* nx +i]) / (2 * dy) * 
        //                              (v[j* nx +i+1] - v[j* nx +i-1]) / (2 * dx)) - 
        //                         pow((v[(j+1)* nx +i] - v[(j-1)* nx +i]) / (2 * dy), 2));
        //     }
        // }
        build_up_p<<<(nx*ny+nx-1)/nx, nx>>>(d_p,d_p1, d_pn, d_b, nx, ny, dt, dx, dy);
        CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);
        // for (int it = 0; it < nit; it++) { 
        //     pn = p;
        //     for (int j = 1; j < ny-1; j++) {
        //         for (int i = 1; i < nx-1; i++) {
        //             p[j * nx + i] = (dy*dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]) +
        //                              dx*dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]) -
        //                              b[j * nx + i] * dx*dx * dy*dy) / 
        //                             (2 * (dx*dx + dy*dy));
        //         }
        //     }
        //     for (int j = 0; j < ny; j++) p[j * nx + nx - 1] = p[j * nx + nx - 2];
        //     for (int i = 0; i < nx; i++) p[i] = p[nx + i];
        //     for (int j = 0; j < ny; j++) p[j * nx] = p[j * nx + 1];
        //     for (int i = 0; i < nx; i++) p[(ny - 1) * nx + i] = 0;
        // }
        
        build_up_unvn<<<(nx*ny+nx-1)/nx, nx>>>(d_un,d_u, d_vn, d_v, nx, ny);
        CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);
        // Matrix un = u;
        // Matrix vn = v;
        build_up_uv<<<(nx*ny+nx-1)/nx, nx>>>(d_u, d_un, d_v, d_vn,d_p, nx, ny,rho,  dt,  dx,  dy,nu);
        CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);
        // for (int j = 1; j < ny-1; j++) {
        //     for (int i = 1; i < nx-1; i++) {
        //         u[j * nx + i] = un[j * nx + i] - un[j * nx + i] * dt / dx * (un[j * nx + i] - un[j * nx + i - 1]) -
        //                   un[j * nx + i] * dt / dy * (un[j * nx + i] - un[(j - 1) * nx + i]) -
        //                   dt / (2 * rho * dx) * (p[j * nx + i + 1] - p[j * nx + i - 1]) +
        //                   nu * dt / (dx*dx) * (un[j * nx + i + 1] - 2 * un[j * nx + i] + un[j * nx + i - 1]) +
        //                   nu * dt / (dy*dy) * (un[(j + 1) * nx + i] - 2 * un[j * nx + i] + un[(j - 1) * nx + i]);
        //         v[j * nx + i] = vn[j * nx + i] - vn[j * nx + i] * dt / dx * (vn[j * nx + i] - vn[j * nx + i - 1]) -
        //                   vn[j * nx + i] * dt / dy * (vn[j * nx + i] - vn[(j - 1) * nx + i]) -
        //                   dt / (2 * rho * dx) * (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) +
        //                   nu * dt / (dx*dx) * (vn[j * nx + i + 1] - 2 * vn[j * nx + i] + vn[j * nx + i - 1]) +
        //                   nu * dt / (dy*dy) * (vn[(j + 1) * nx + i] - 2 * vn[j * nx + i] + vn[(j - 1) * nx + i]);
        //     }
        // }

        // for (int i = 0; i < nx; i++) u[i] = 0;
        // for (int j = 0; j < ny; j++) u[j * nx] = 0;
        // for (int j = 0; j < ny; j++) u[j * nx + nx - 1] = 0;
        // for (int i = 0; i < nx; i++) u[(ny - 1) * nx + i] = 1;
        // for (int i = 0; i < nx; i++) v[i] = 0;
        // for (int i = 0; i < nx; i++) v[(ny - 1) * nx + i] = 0;
        // for (int j = 0; j < ny; j++) v[j * nx] = 0;
        // for (int j = 0; j < ny; j++) v[j * nx + nx - 1] = 0;

        if (n % 10 == 0) {
            CHECK_ERROR(cudaMemcpy(&u[0],d_u, nx * ny * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
            CHECK_ERROR(cudaMemcpy(&v[0],d_v,  nx * ny * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
            CHECK_ERROR(cudaMemcpy(&p[0],d_p,  nx * ny * sizeof(double), cudaMemcpyDeviceToHost), __FILE__, __LINE__);
            CHECK_ERROR(cudaDeviceSynchronize(), __FILE__, __LINE__);
            write_csv("u_" + std::to_string(n) + ".csv", u, ny, nx);
            write_csv("v_" + std::to_string(n) + ".csv", v, ny, nx);
            write_csv("p_" + std::to_string(n) + ".csv", p, ny, nx);
        }
    }

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);

    return 0;
}
