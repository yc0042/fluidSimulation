#include <SDL.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <thread>
#undef main
using namespace std::chrono_literals;

#define IX(x, y)  ((x)+(y)*N)

struct FluidCell {
    int size;
    float dt;
    float diff;
    float visc;

    float* s;
    float* density;

    float* Vx;
    float* Vy;

    float* Vx0;
    float* Vy0;
};

typedef struct FluidCell FluidCell;


FluidCell* FluidCellCreate(int size, float diffusion, float viscosity, float dt) {
    FluidCell* cell = (FluidCell*)malloc(sizeof(*cell));
    int N = size;


    cell->size = size;
    cell->dt = dt;
    cell->diff = diffusion;
    cell->visc = viscosity;

    cell->s = (float*)calloc(N * N, sizeof(float));
    cell->density = (float*)calloc(N * N, sizeof(float));

    cell->Vx = (float*)calloc(N * N, sizeof(float));
    cell->Vy = (float*)calloc(N * N, sizeof(float));

    cell->Vx0 = (float*)calloc(N * N, sizeof(float));
    cell->Vy0 = (float*)calloc(N * N, sizeof(float));

    return cell;
}

void FluidCellFree(FluidCell* cell)
{
    free(cell->s);
    free(cell->density);

    free(cell->Vx);
    free(cell->Vy);

    free(cell->Vx0);
    free(cell->Vy0);

    free(cell);
}

void FluidCellAddDensity(FluidCell* cell, int x, int y, float amount)
{
    int N = cell->size;
    cell->density[IX(x, y)] += amount;
}

void FluidCellAddVelocity(FluidCell* cell, int x, int y, float amountX, float amountY)
{
    int N = cell->size;
    int index = IX(x, y);

    cell->Vx[index] += amountX;
    cell->Vy[index] += amountY;
}

static void set_bnd(int b, float* x, int N)
{
    for (int i = 1; i < N - 1; i++) {
        x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, N - 1)] = b == 2 ? -x[IX(i, N - 2)] : x[IX(i, N - 2)];
    }
    for (int j = 1; j < N - 1; j++) {
        x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
        x[IX(N - 1, j)] = b == 1 ? -x[IX(N - 2, j)] : x[IX(N - 2, j)];
    }

    x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, N - 1)] = 0.5 * (x[IX(1, N - 1)] + x[IX(0, N - 2)]);
    x[IX(N - 1, 0)] = 0.5 * (x[IX(N - 2, 0)] + x[IX(N - 1, 1)]);
    x[IX(N - 1, N - 1)] = 0.5 * (x[IX(N - 2, N - 1)] + x[IX(N - 1, N - 2)]);
}

static void lin_solve(int b, float* x, float* x0, float a, float c, int iter, int N) {
    float cRecip = 1.0 / c;
    for (int t = 0; t < iter; t++) {
        for (int j = 1; j < N - 1; j++) {
            for (int i = 1; i < N - 1; i++) {
                x[IX(i, j)] =
                    (x0[IX(i, j)] +
                        a *
                        (x[IX(i + 1, j)] +
                            x[IX(i - 1, j)] +
                            x[IX(i, j + 1)] +
                            x[IX(i, j - 1)])) *
                    cRecip;
            }
        }
        set_bnd(b, x, N);
    }
}

static void diffuse(int b, float* x, float* x0, float diff, float dt, int iter, int N)
{
    float a = dt * diff * (N - 2) * (N - 2);
    lin_solve(b, x, x0, a, 1 + 6 * a, iter, N);
}

static void project(float* velocX, float* velocY, float* p, float* div, int iter, int N) {
    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            div[IX(i, j)] =
                (-0.5 *
                    (velocX[IX(i + 1, j)] -
                        velocX[IX(i - 1, j)] +
                        velocY[IX(i, j + 1)] -
                        velocY[IX(i, j - 1)])) /
                N;
            p[IX(i, j)] = 0;
        }
    }

    set_bnd(0, div, N);
    set_bnd(0, p, N);
    lin_solve(0, p, div, 1, 6, iter, N);

    for (int j = 1; j < N - 1; j++) {
        for (int i = 1; i < N - 1; i++) {
            velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * N;
            velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * N;
        }
    }

    set_bnd(1, velocX, N);
    set_bnd(2, velocY, N);
}

static void advect(int b, float* d, float* d0, float* velocX, float* velocY, float dt, int N) {
    float i0, i1, j0, j1;

    float dtx = dt * (N - 2);
    float dty = dt * (N - 2);

    float s0, s1, t0, t1;
    float tmp1, tmp2, x, y;

    float Nfloat = N;
    float ifloat, jfloat;
    int i, j;

    for (j = 1, jfloat = 1; j < N-1; j++, jfloat++) {
        for (i = 1, ifloat = 1; i < N-1; i++, ifloat++) {
            tmp1 = dtx * velocX[IX(i, j)];
            tmp2 = dty * velocY[IX(i, j)];
            x = ifloat - tmp1;
            y = jfloat - tmp2;

            if (x < 0.5) x = 0.5;
            if (x > Nfloat) x = Nfloat;
            i0 = floor(x);
            i1 = i0 + 1.0;
            if (y < 0.5) y = 0.5;
            if (y > Nfloat) y = Nfloat;
            j0 = floor(y);
            j1 = j0 + 1.0;

            s1 = x - i0;
            s0 = 1.0 - s1;
            t1 = y - j0;
            t0 = 1.0 - t1;

            int i0i = (int)i0;
            int i1i = (int)i1;
            int j0i = (int)j0;
            int j1i = (int)j1;

            d[IX(i, j)] =
                s0 * (t0 * d0[IX(i0i, j0i)] + t1 * d0[IX(i0i, j1i)]) +
                s1 * (t0 * d0[IX(i1i, j0i)] + t1 * d0[IX(i1i, j1i)]);
        }
    }

    set_bnd(b, d, N);
}

void FluidCellStep(FluidCell* cell)
{
    int N = cell->size;
    float visc = cell->visc;
    float diff = cell->diff;

    float dt = cell->dt;
    float* Vx = cell->Vx;
    float* Vy = cell->Vy;

    float* Vx0 = cell->Vx0;
    float* Vy0 = cell->Vy0;

    float* s = cell->s;
    float* density = cell->density;

    diffuse(1, Vx0, Vx, visc, dt, 4, N);
    diffuse(2, Vy0, Vy, visc, dt, 4, N);

    project(Vx0, Vy0, Vx, Vy, 4, N);

    advect(1, Vx, Vx0, Vx0, Vy0, dt, N);
    advect(2, Vy, Vy0, Vx0, Vy0, dt, N);

    project(Vx, Vy, Vx0, Vy0, 4, N);

    diffuse(0, s, density, diff, dt, 4, N);
    advect(0, density, s, Vx, Vy, dt, N);
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow("Pixel Grid", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 1440, 1440, SDL_WINDOW_SHOWN);
    if (window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == NULL) {
        printf("Renderer could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }
    
    int N = 160;

    // Create a surface to hold the pixel data
    const int width = 160;
    const int height = 160;
    int depth = 32;
    Uint32 rmask, gmask, bmask, amask;
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    rmask = 0xff000000;
    gmask = 0x00ff0000;
    bmask = 0x0000ff00;
    amask = 0x000000ff;
#else
    rmask = 0x000000ff;
    gmask = 0x0000ff00;
    bmask = 0x00ff0000;
    amask = 0xff000000;
#endif
    SDL_Surface* surface = SDL_CreateRGBSurface(0, width, height, depth, rmask, gmask, bmask, amask);
    if (surface == NULL) {
        printf("Surface could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    // Initialize Fluid Cell
    FluidCell* cell = FluidCellCreate(160, 0.00000005f, 0.000000001f, 0.033f);

    // Fill the surface with the pixel data from the 2D-array
    float* pixel_grid = cell->density;
    // fill pixel_grid with color values
    SDL_LockSurface(surface);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Uint32* pixel = (Uint32*)((Uint8*)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
            *pixel = SDL_MapRGBA(surface->format, *(pixel_grid + IX(x,y)), *(pixel_grid + IX(x, y)), *(pixel_grid + IX(x, y)), 255);
        }
    }
    SDL_UnlockSurface(surface);

    //// Create a texture from the surface
    SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (texture == NULL) {
        printf("Texture could not be created! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }
    // Clear the renderer
    SDL_RenderClear(renderer);

    // Render the texture on the renderer
    SDL_RenderCopy(renderer, texture, NULL, NULL);

    // Update the screen
    SDL_RenderPresent(renderer);
    
    // Wait for user to close the window
    bool quit = false;
    SDL_Event e;
    int x0, xf, y0, yf;
    x0 = 48;
    y0 = 48;
    bool held = false;
    int iter_count = 0;
    while (!quit) {
        auto start = std::chrono::steady_clock::now();
        
        if (iter_count % 4 == 0)
        {
            SDL_LockSurface(surface);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    Uint32* pixel = (Uint32*)((Uint8*)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
                    *pixel = SDL_MapRGBA(surface->format, std::min(*(pixel_grid + IX(x, y)), 255.0f), std::min(*(pixel_grid + IX(x, y)), 255.0f), std::min(*(pixel_grid + IX(x, y)), 255.0f), 255);
                }
            }
            SDL_UnlockSurface(surface);

            texture = SDL_CreateTextureFromSurface(renderer, surface);

            SDL_RenderClear(renderer);
            SDL_RenderCopy(renderer, texture, NULL, NULL);
            SDL_RenderPresent(renderer);
        }
        
        auto end = std::chrono::steady_clock::now();

        SDL_GetMouseState(&xf, &yf);
        xf /= 9;
        yf /= 9;

        if (xf < 0) xf = 0;
        if (yf < 0) yf = 0;

        std::cout << xf << " : " << yf << std::endl;


        if (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_MOUSEBUTTONDOWN)
            {
                if (SDL_BUTTON_LEFT == e.button.button)
                {
                    std::cout << "Mouse Down!" << std::endl;
                    held = true;
                    
                }
            }
            if (e.type == SDL_MOUSEBUTTONUP)
            {
                if (SDL_BUTTON_LEFT == e.button.button)
                {
                    std::cout << "Mouse Released!" << std::endl;
                    held = false;
                }
            }
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        if (held)
        {
            FluidCellAddDensity(cell, xf, yf, 90.0f);
            FluidCellAddDensity(cell, xf+1, yf+1, 90.0f);
            FluidCellAddDensity(cell, xf+1, yf, 90.0f);
            FluidCellAddDensity(cell, xf, yf+1, 90.0f);
            FluidCellAddVelocity(cell, xf, yf, (xf - x0) * 0.5f, (yf - y0) * 0.5f);
        }

        FluidCellStep(cell);
        auto elapsed = end - start;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        std::cout << "simulation frame took " << elapsed_ms.count() << "ms" << std::endl;
        auto sleep_duration = 8.33ms - elapsed;

        std::this_thread::sleep_for(sleep_duration);
        x0 = xf;
        y0 = yf;
    }

    // Clean up
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
