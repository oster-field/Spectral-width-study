"""Construction and storage of the distribution function of amplitudes/local extrema of a wave field with
different spectra."""
from main import WaveFieldSimulation

if __name__ == '__main__':
    simulation = WaveFieldSimulation(
        num_realizations=200,
        max_w=900,
        num_harmonics=2 ** 13,
        spectrum_w0=17.5,
        power_spectrum=6,
        a=1,
        b=5,
        kc=1e5,
        k=1e-5,
        ampl_or_extr='amplitude',
        name_of_spectrum='Gaussian',
        showcase=False
    )
    simulation.run_simulation()
