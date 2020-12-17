#!/usr/bin/env python3

# labsynth
# Additive synthesizer for MILAB2 university course
# Copyright (c) 2020 Alexander F. Mayer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import jack
import numpy as np
import threading

# constants
MIDI_CMD_NOTEOFF = 8
MIDI_CMD_NOTEON = 9
MIDI_CMD_CC = 11
TIME_CONTROL_BASE = 1.03
MIN_RAMP_TIME_SEC = 0.005
MAX_ATTACK_TIME_SEC = 1.0
MAX_DECAY_TIME_SEC = 1.0
MAX_RELEASE_TIME_SEC = 1.0

# argument parsing
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--jackname",
                        default="labsynth",
                        help="set JACK client name",
                        metavar="NAME")
arg_parser.add_argument("--cc-volume",
                        default=7,
                        help="set MIDI CC for volume control",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-pan",
                        default=10,
                        help="set MIDI CC for pan",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-attack",
                        default=20,
                        help="set MIDI CC for attack time",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-decay",
                        default=21,
                        help="set MIDI CC for decay time",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-sustain",
                        default=22,
                        help="set MIDI CC for sustain level",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-release",
                        default=23,
                        help="set MIDI CC for release time",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--cc-partials",
                        default=24,
                        help="set MIDI CC for number of partials per voice",
                        metavar="CC_INDEX",
                        type=int)
arg_parser.add_argument("--tune",
                        default=440,
                        help="set frequency of A4 note",
                        metavar="FREQUENCY",
                        type=float)
arg_parser.add_argument("--dump-midi",
                        help="enable MIDI dump",
                        action="store_true")
arg_parser.add_argument("--dump-parameters",
                        help="continuously show parameters on console",
                        action="store_true")
args = arg_parser.parse_args()

# initialize globals: JACK client stuff
jack_sr = 0
jack_client = jack.Client(args.jackname)
midi_in_port = jack_client.midi_inports.register("midi-in")
audio_out_port_l = jack_client.outports.register("out-left")
audio_out_port_r = jack_client.outports.register("out-right")

# initialize globals: event loop
event = threading.Event()

# initialize globals: voices
voice_data = {}

# initialize globals: sound parameters
param_volume = 1.0
param_pan_left = 1.0
param_pan_right = 1.0
param_attack_time = 0.0
param_decay_time = 0.0
param_sustain_level = 1.0
param_release_time = 0.0
param_num_of_partials = 1

def dump_parameters():
    """Clear console screen and dump human-readable parameters."""
    print("\x1b[2J\x1b[H")
    print("   Volume: {}\n    Pan-L: {}\n    Pan-R: {}\n".format(param_volume,
                param_pan_left, param_pan_right))
    print("   Attack: {}\n    Decay: {}\n  Sustain: {}\n  Release: {}\n".format(
                param_attack_time, param_decay_time, param_sustain_level,
                param_release_time))
    print(" Partials: {}\n".format(param_num_of_partials))

def m2f(midi_note):
    """Return the frequency (in Hz) for a given MIDI note value."""
    return 2 ** ((midi_note - 69) / 12) * args.tune

def set_cc(cc, value):
    """Change audio parameter value according to CC mappings."""
    global param_volume, param_pan_left, param_pan_right, param_attack_time
    global param_decay_time, param_sustain_level, param_release_time
    global param_num_of_partials
    if value >= 128:
        return
    if cc == args.cc_volume:
        param_volume = value / 100 if value != 0 else 0.0
    elif cc == args.cc_pan:
        # HACK: it's not a real pan
        if value == 63:
            param_pan_left = 1.0
            param_pan_right = 1.0
        elif value < 63:
            param_pan_left = 1.0
            param_pan_right = 1.0 - (63 - value) / 64 if value != 0 else 0.0
        else:
            param_pan_left = 1.0 - (value - 63) / 64
            param_pan_right = 1.0
    elif cc == args.cc_attack:
        param_attack_time = (((TIME_CONTROL_BASE ** value) - 1) /
                ((TIME_CONTROL_BASE ** 127) - 1)) * (MAX_ATTACK_TIME_SEC -
                MIN_RAMP_TIME_SEC) + MIN_RAMP_TIME_SEC
    elif cc == args.cc_decay:
        param_decay_time = (((TIME_CONTROL_BASE ** value) - 1) /
                ((TIME_CONTROL_BASE ** 127) - 1)) * (MAX_DECAY_TIME_SEC -
                MIN_RAMP_TIME_SEC) + MIN_RAMP_TIME_SEC
    elif cc == args.cc_sustain:
        param_sustain_level = value / 127 if value != 0 else 0.0
    elif cc == args.cc_release:
        param_release_time = (((TIME_CONTROL_BASE ** value) - 1) /
                ((TIME_CONTROL_BASE ** 127) - 1)) * (MAX_RELEASE_TIME_SEC -
                MIN_RAMP_TIME_SEC) + MIN_RAMP_TIME_SEC
    elif cc == args.cc_partials:
        # minimum is 1 voice
        param_num_of_partials = value if value != 0 else 1

    if args.dump_parameters:
        dump_parameters()

def generate_ads_envelope(env_array, start_idx, velocity, attack_samples,
                          decay_samples, sustain_factor, slope_start_value):
    """Generate an envelope for attack, decay and sustain phase of ADSR.

    Arguments:
    env_array      - array for envelope values (changed in-place)
    start_idx      - number of samples since start of envelope
    velocity       - maximum velocity after attack (0.0 - 1.0)
    attack_samples - number of samples before maximum velocity is reached
    decay_samples  - number of samples for decay time
    sustain_factor - sustain level relative to maximum velocity (0.0 - 1.0)
    slope_start_value - assume value at start_idx and continue slope from there
    """
    pos = 0
    blocksize = len(env_array)
    while pos < blocksize:
        remaining = blocksize - pos
        current_offset = start_idx + pos
        if current_offset < attack_samples:
            # attack phase
            slope = np.linspace(slope_start_value, velocity, num=attack_samples)
            num_samples = min(attack_samples - current_offset, remaining)
            first_idx = current_offset
            last_idx = current_offset + num_samples
            env_array[pos:pos+num_samples] = slope[first_idx:last_idx]
        elif current_offset >= attack_samples and current_offset < (
                        attack_samples + decay_samples):
            # decay phase
            slope = np.linspace(velocity, velocity * sustain_factor,
                                num=decay_samples)
            num_samples = min(attack_samples + decay_samples - current_offset,
                              remaining)
            first_idx = current_offset - attack_samples
            last_idx = current_offset + num_samples - attack_samples
            env_array[pos:pos+num_samples] = slope[first_idx:last_idx]
        else:
            # sustain phase
            num_samples = remaining
            env_array[pos:pos+num_samples] = velocity * sustain_factor
        pos = pos + num_samples

def generate_release_envelope(env_array, start_idx, note_off_idx,
                              release_samples, slope_start_value):
    """Generate an envelope for the release phase of ADSR.

    Arguments:
    env_array       - array for envelope values (changed in-place)
    start_idx       - number of samples since start of envelope
    note_off_idx    - number of samples since NOTE_OFF
    release_samples - number of samples for release time
    slope_start_value - assume value at start_idx and continue slope from there
    """
    completed = False
    pos = 0
    blocksize = len(env_array)
    while pos < blocksize:
        remaining = blocksize - pos
        current_offset = start_idx + pos
        if note_off_idx < release_samples:
            # release phase
            slope = np.linspace(slope_start_value, 0.0, num=release_samples)
            num_samples = min(release_samples - note_off_idx, remaining)
            first_idx = note_off_idx
            last_idx = note_off_idx + num_samples
            env_array[pos:pos+num_samples] = slope[first_idx:last_idx]
        else:
            completed = True
            num_samples = remaining
            env_array[pos:pos+num_samples] = 0
        pos = pos + num_samples
    return completed

def render_voice(buffer_array, midi_pitch, times, partial_fn):
    """Render one voice with all its partials."""
    buf = np.zeros(len(buffer_array))
    for n in range(1, param_num_of_partials + 1):
        np.sin(2 * np.pi * m2f(midi_pitch) * n * times, out=buf)
        buffer_array += buf * partial_fn(n)

@jack_client.set_samplerate_callback
def samplerate(samplerate):
    """Adapt some internal variables to current sample rate, reset voices."""
    global jack_sr
    jack_sr = samplerate
    voice_data.clear()

@jack_client.set_process_callback
def jack_process(jack_blocksize):
    """Do the work to be done in the JACK process callback."""
    now_samples = jack_client.last_frame_time

    # process incoming MIDI messages (notes, CC)
    for time, midi_data in midi_in_port.incoming_midi_events():
        if args.dump_midi:
            print("Incoming MIDI: {}".format(bytes(midi_data).hex(" ")))

        # handle only MIDI messages with a length of 3 bytes
        if len(midi_data) != 3:
            continue

        midi_bytes = bytes(midi_data)
        command = midi_bytes[0] >> 4
        midi_pitch = midi_bytes[1]
        velocity = midi_bytes[2]
        if command == MIDI_CMD_CC:
            set_cc(midi_bytes[1], midi_bytes[2])
        elif command == MIDI_CMD_NOTEON and velocity != 0:
            try:
                env_array, _, _, slope_start_value, _ = voice_data[midi_pitch]
            except KeyError:
                env_array = np.zeros(jack_blocksize)
                slope_start_value = 0.0
            voice_data[midi_pitch] = (env_array, now_samples, -1, env_array[-1],
                                      velocity / 127)
        elif command == MIDI_CMD_NOTEOFF or command == MIDI_CMD_NOTEON:
            try:
                env_array, start_time, _, slope_start_value, old_velocity = (
                            voice_data[midi_pitch])
            except KeyError:
                continue
            # store current time (NOTE_OFF event)
            voice_data[midi_pitch] = (env_array, start_time, now_samples,
                                      env_array[-1], old_velocity)

    # generate audio
    buffer_left = audio_out_port_l.get_array()
    buffer_right = audio_out_port_r.get_array()
    buffer_left.fill(0)
    buffer_right.fill(0)
    render_array = np.zeros(jack_blocksize)
    times = np.arange(now_samples, now_samples + jack_blocksize) / jack_sr
    attack_samples = int(jack_sr * param_attack_time)
    decay_samples = int(jack_sr * param_decay_time)
    release_samples = int(jack_sr * param_release_time)

    for midi_pitch, (env_array, start_time, note_off_time, slope_start_value,
                     velocity) in voice_data.items():
        if note_off_time < 0:
            # voice is in ADS phase of ADSR
            generate_ads_envelope(env_array, now_samples - start_time,
                                  velocity, attack_samples, decay_samples,
                                  param_sustain_level, slope_start_value)
        else:
            # voice is in release phase of ADSR
            completed = generate_release_envelope(env_array,
                                                  now_samples - start_time,
                                                  now_samples - note_off_time,
                                                  release_samples,
                                                  slope_start_value)
            if completed:
                # store a velocity of 0 to indicate this voice can be deleted
                voice_data[midi_pitch] = (env_array, start_time, note_off_time,
                                          slope_start_value, 0)

        render_voice(render_array, midi_pitch, times - start_time / jack_sr,
                     lambda x: (1 / x**2))
        buffer_left += render_array * env_array * param_pan_left * param_volume
        buffer_right += (render_array * env_array * param_pan_right *
                         param_volume)

    # remove outdated voices
    for midi_pitch in list(voice_data):
        _, _, _, _, velocity = voice_data[midi_pitch]
        if velocity == 0:
            del voice_data[midi_pitch]

@jack_client.set_shutdown_callback
def jack_shutdown(status, reason):
    """Set the flag to exit the event loop."""
    print("JACK shutdown: {} ({})".format(reason, status))
    event.set()

# run event loop (this blocks in the wait function)
with jack_client:
    try:
        event.wait()
    except KeyboardInterrupt:
        print("\nInterrupted")
