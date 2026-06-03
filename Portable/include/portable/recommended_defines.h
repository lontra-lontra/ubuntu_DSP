#pragma once

// Recommended local defaults for audio-focused apps.
// This header is intentionally not referenced anywhere yet.

#ifndef DEVICE_NAME
#define DEVICE_NAME "MADIface USB (24285073): Audio (hw:1,0)"
#endif

#ifndef CHANNELS
#define CHANNELS 32
#endif

#ifndef FRAMES_PER_BUFFER
#define FRAMES_PER_BUFFER 256
#endif

#ifndef SAMPLE_FORMAT
#define SAMPLE_FORMAT paFloat32
#endif

#ifndef SAMPLE_RATE
#define SAMPLE_RATE 48000
#endif
