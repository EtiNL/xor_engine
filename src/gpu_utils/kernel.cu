extern "C" __global__ void generate_image(int width, int height, int mouse_x, int mouse_y, unsigned char *image) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = 20; // radius of the circle

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int dx = x - mouse_x;
        int dy = y - mouse_y;
        if (dx * dx + dy * dy <= radius * radius) {
            // Inside the circle: set to red
            image[idx] = 255;     // Red channel
            image[idx + 1] = 0;   // Green channel
            image[idx + 2] = 0;   // Blue channel
        } else {
            // Outside the circle: original gradient
            image[idx] = x % 256;     // Red channel
            image[idx + 1] = y % 256; // Green channel
            image[idx + 2] = (x + y) % 256; // Blue channel
        }
    }
}
