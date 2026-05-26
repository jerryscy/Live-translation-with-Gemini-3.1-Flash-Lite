/**
 * AudioWorklet that converts the browser's 32-bit float microphone samples
 * into 16-bit little-endian PCM frames and posts them back to the main
 * thread, which forwards them to the FastAPI server over the WebSocket.
 *
 * The AudioContext is created at 16 kHz, so no resampling is needed here.
 */
class AudioProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input.length > 0 && input[0] && input[0].length > 0) {
            const channel = input[0];
            const pcm = new Int16Array(channel.length);
            for (let i = 0; i < channel.length; i++) {
                let s = Math.max(-1, Math.min(1, channel[i]));
                pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            }
            this.port.postMessage(pcm.buffer, [pcm.buffer]);
        }
        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
