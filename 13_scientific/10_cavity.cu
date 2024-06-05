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

typedef std::vector<std::vector<double>> Matrix;

Matrix zeros(int rows, int cols) {
    return Matrix(rows, std::vector<double>(cols, 0.0));
}

void write_csv(const std::string &filename, const Matrix &data) {
    std::ofstream file(filename);
    for (const auto& row : data) {
        for (size_t col = 0; col < row.size(); ++col) {
            file << row[col];
            if (col != row.size() - 1) file << ",";
        }
        file << "\n";
    }
}

int main() {
    Matrix u = zeros(ny, nx);
    Matrix v = zeros(ny, nx);
    Matrix p = zeros(ny, nx);
    Matrix b = zeros(ny, nx);

    for (int n = 0; n < nt; n++) {
        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                b[j][i] = rho * (1 / dt * 
                                ((u[j][i+1] - u[j][i-1]) / (2 * dx) + 
                                 (v[j+1][i] - v[j-1][i]) / (2 * dy)) - 
                                pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) - 
                                2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * 
                                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - 
                                pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));
            }
        }

        for (int it = 0; it < nit; it++) {
            Matrix pn = p;
            for (int j = 1; j < ny-1; j++) {
                for (int i = 1; i < nx-1; i++) {
                    p[j][i] = (dy*dy * (pn[j][i+1] + pn[j][i-1]) +
                               dx*dx * (pn[j+1][i] + pn[j-1][i]) -
                               b[j][i] * dx*dx * dy*dy) / 
                              (2 * (dx*dx + dy*dy));
                }
            }
            for (int j = 0; j < ny; j++) p[j][nx-1] = p[j][nx-2];
            for (int i = 0; i < nx; i++) p[0][i] = p[1][i];
            for (int j = 0; j < ny; j++) p[j][0] = p[j][1];
            for (int i = 0; i < nx; i++) p[ny-1][i] = 0;
        }

        Matrix un = u;
        Matrix vn = v;

        for (int j = 1; j < ny-1; j++) {
            for (int i = 1; i < nx-1; i++) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1]) -
                          un[j][i] * dt / dy * (un[j][i] - un[j-1][i]) -
                          dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1]) +
                          nu * dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1]) +
                          nu * dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1]) -
                          vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i]) -
                          dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i]) +
                          nu * dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1]) +
                          nu * dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }

        for (int i = 0; i < nx; i++) u[0][i] = 0;
        for (int j = 0; j < ny; j++) u[j][0] = 0;
        for (int j = 0; j < ny; j++) u[j][nx-1] = 0;
        for (int i = 0; i < nx; i++) u[ny-1][i] = 1;
        for (int i = 0; i < nx; i++) v[0][i] = 0;
        for (int i = 0; i < nx; i++) v[ny-1][i] = 0;
        for (int j = 0; j < ny; j++) v[j][0] = 0;
        for (int j = 0; j < ny; j++) v[j][nx-1] = 0;

        if (n % 10 == 0) {
            write_csv("u_" + std::to_string(n) + ".csv", u);
            write_csv("v_" + std::to_string(n) + ".csv", v);
            write_csv("p_" + std::to_string(n) + ".csv", p);
        }
    }

    return 0;
}
