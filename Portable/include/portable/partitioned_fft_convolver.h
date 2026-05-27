#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include <fftw3.h>

struct PartitionedFftConvolver
{
    int block_size = 0;
    int fft_size = 0;
    int fft_bins = 0;
    int partition_count = 0;
    int write_partition = 0;
    float scale = 1.0f;
    bool spectra_precomputed = false;

    std::vector<float> h_re;
    std::vector<float> h_im;
    std::vector<float> x_re;
    std::vector<float> x_im;
    std::vector<float> y_re;
    std::vector<float> y_im;
    std::vector<float> overlap_tail;

    float *time_in = nullptr;
    float *time_out = nullptr;
    fftwf_complex *freq = nullptr;
    fftwf_plan forward_plan = nullptr;
    fftwf_plan inverse_plan = nullptr;

    PartitionedFftConvolver() = default;
    PartitionedFftConvolver(const PartitionedFftConvolver &) = delete;
    PartitionedFftConvolver &operator=(const PartitionedFftConvolver &) = delete;
    PartitionedFftConvolver(PartitionedFftConvolver &&other) noexcept
    {
        *this = std::move(other);
    }

    PartitionedFftConvolver &operator=(PartitionedFftConvolver &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        release();

        block_size = other.block_size;
        fft_size = other.fft_size;
        fft_bins = other.fft_bins;
        partition_count = other.partition_count;
        write_partition = other.write_partition;
        scale = other.scale;
        spectra_precomputed = other.spectra_precomputed;

        h_re = std::move(other.h_re);
        h_im = std::move(other.h_im);
        x_re = std::move(other.x_re);
        x_im = std::move(other.x_im);
        y_re = std::move(other.y_re);
        y_im = std::move(other.y_im);
        overlap_tail = std::move(other.overlap_tail);

        time_in = other.time_in;
        time_out = other.time_out;
        freq = other.freq;
        forward_plan = other.forward_plan;
        inverse_plan = other.inverse_plan;

        other.time_in = nullptr;
        other.time_out = nullptr;
        other.freq = nullptr;
        other.forward_plan = nullptr;
        other.inverse_plan = nullptr;
        other.block_size = 0;
        other.fft_size = 0;
        other.fft_bins = 0;
        other.partition_count = 0;
        other.write_partition = 0;
        other.scale = 1.0f;
        other.spectra_precomputed = false;

        return *this;
    }

    ~PartitionedFftConvolver()
    {
        release();
    }

    void release()
    {
        if (forward_plan)
        {
            fftwf_destroy_plan(forward_plan);
            forward_plan = nullptr;
        }
        if (inverse_plan)
        {
            fftwf_destroy_plan(inverse_plan);
            inverse_plan = nullptr;
        }
        if (time_in)
        {
            fftwf_free(time_in);
            time_in = nullptr;
        }
        if (time_out)
        {
            fftwf_free(time_out);
            time_out = nullptr;
        }
        if (freq)
        {
            fftwf_free(freq);
            freq = nullptr;
        }

        block_size = 0;
        fft_size = 0;
        fft_bins = 0;
        partition_count = 0;
        write_partition = 0;
        scale = 1.0f;
        spectra_precomputed = false;
        h_re.clear();
        h_im.clear();
        x_re.clear();
        x_im.clear();
        y_re.clear();
        y_im.clear();
        overlap_tail.clear();
    }

    bool initialize(const std::vector<float> &kernel, int block_size_samples)
    {
        release();

        if (kernel.empty() || block_size_samples <= 0)
        {
            return false;
        }

        block_size = block_size_samples;
        fft_size = 2 * block_size;
        fft_bins = fft_size / 2 + 1;
        partition_count =
            static_cast<int>(
                (kernel.size() + static_cast<size_t>(block_size) - 1) /
                static_cast<size_t>(block_size));
        if (partition_count <= 0)
        {
            return false;
        }

        scale = 1.0f / static_cast<float>(fft_size);

        time_in = static_cast<float *>(fftwf_malloc(sizeof(float) * fft_size));
        time_out = static_cast<float *>(fftwf_malloc(sizeof(float) * fft_size));
        freq = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * fft_bins));
        if (!time_in || !time_out || !freq)
        {
            release();
            return false;
        }

        std::fill(time_in, time_in + fft_size, 0.0f);
        std::fill(time_out, time_out + fft_size, 0.0f);
        for (int k = 0; k < fft_bins; ++k)
        {
            freq[k][0] = 0.0f;
            freq[k][1] = 0.0f;
        }

        forward_plan = fftwf_plan_dft_r2c_1d(fft_size, time_in, freq, FFTW_MEASURE);
        inverse_plan = fftwf_plan_dft_c2r_1d(fft_size, freq, time_out, FFTW_MEASURE);
        if (!forward_plan || !inverse_plan)
        {
            release();
            return false;
        }

        const size_t spectral_size =
            static_cast<size_t>(partition_count) * static_cast<size_t>(fft_bins);
        h_re.assign(spectral_size, 0.0f);
        h_im.assign(spectral_size, 0.0f);
        x_re.assign(spectral_size, 0.0f);
        x_im.assign(spectral_size, 0.0f);
        y_re.assign(static_cast<size_t>(fft_bins), 0.0f);
        y_im.assign(static_cast<size_t>(fft_bins), 0.0f);
        overlap_tail.assign(static_cast<size_t>(block_size), 0.0f);

        for (int p = 0; p < partition_count; ++p)
        {
            std::fill(time_in, time_in + fft_size, 0.0f);
            const int start = p * block_size;
            for (int i = 0; i < block_size; ++i)
            {
                const int idx = start + i;
                if (idx >= static_cast<int>(kernel.size()))
                {
                    break;
                }
                time_in[i] = kernel[static_cast<size_t>(idx)];
            }

            fftwf_execute(forward_plan);

            const size_t base =
                static_cast<size_t>(p) * static_cast<size_t>(fft_bins);
            for (int k = 0; k < fft_bins; ++k)
            {
                h_re[base + static_cast<size_t>(k)] = freq[k][0];
                h_im[base + static_cast<size_t>(k)] = freq[k][1];
            }
        }

        spectra_precomputed = true;
        reset_state();
        return true;
    }

    void reset_state()
    {
        std::fill(x_re.begin(), x_re.end(), 0.0f);
        std::fill(x_im.begin(), x_im.end(), 0.0f);
        std::fill(overlap_tail.begin(), overlap_tail.end(), 0.0f);
        write_partition = partition_count - 1;
    }

    void process_block(
        const std::vector<float> &input_block,
        int input_count,
        std::vector<float> &output_block)
    {
        if (output_block.size() != static_cast<size_t>(block_size))
        {
            output_block.assign(static_cast<size_t>(block_size), 0.0f);
        }

        if (partition_count <= 0 ||
            block_size <= 0 ||
            fft_bins <= 0 ||
            !spectra_precomputed ||
            !forward_plan ||
            !inverse_plan)
        {
            std::fill(output_block.begin(), output_block.end(), 0.0f);
            return;
        }

        input_count = std::max(0, std::min(input_count, block_size));

        std::fill(time_in, time_in + fft_size, 0.0f);
        for (int i = 0; i < input_count; ++i)
        {
            time_in[i] = input_block[static_cast<size_t>(i)];
        }

        fftwf_execute(forward_plan);

        write_partition++;
        if (write_partition >= partition_count)
        {
            write_partition = 0;
        }

        const size_t x_base =
            static_cast<size_t>(write_partition) * static_cast<size_t>(fft_bins);
        for (int k = 0; k < fft_bins; ++k)
        {
            x_re[x_base + static_cast<size_t>(k)] = freq[k][0];
            x_im[x_base + static_cast<size_t>(k)] = freq[k][1];
        }

        std::fill(y_re.begin(), y_re.end(), 0.0f);
        std::fill(y_im.begin(), y_im.end(), 0.0f);

        int x_partition = write_partition;
        for (int p = 0; p < partition_count; ++p)
        {
            const size_t h_base =
                static_cast<size_t>(p) * static_cast<size_t>(fft_bins);
            const size_t xp_base =
                static_cast<size_t>(x_partition) * static_cast<size_t>(fft_bins);

            for (int k = 0; k < fft_bins; ++k)
            {
                const size_t idx_h = h_base + static_cast<size_t>(k);
                const size_t idx_x = xp_base + static_cast<size_t>(k);

                const float hr = h_re[idx_h];
                const float hi = h_im[idx_h];
                const float xr = x_re[idx_x];
                const float xi = x_im[idx_x];

                y_re[static_cast<size_t>(k)] += hr * xr - hi * xi;
                y_im[static_cast<size_t>(k)] += hr * xi + hi * xr;
            }

            x_partition--;
            if (x_partition < 0)
            {
                x_partition = partition_count - 1;
            }
        }

        for (int k = 0; k < fft_bins; ++k)
        {
            freq[k][0] = y_re[static_cast<size_t>(k)];
            freq[k][1] = y_im[static_cast<size_t>(k)];
        }

        fftwf_execute(inverse_plan);

        for (int i = 0; i < block_size; ++i)
        {
            const float y_now =
                time_out[i] * scale + overlap_tail[static_cast<size_t>(i)];
            output_block[static_cast<size_t>(i)] = y_now;
            overlap_tail[static_cast<size_t>(i)] =
                time_out[i + block_size] * scale;
        }
    }
};
