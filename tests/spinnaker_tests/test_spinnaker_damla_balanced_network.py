# -*- coding: utf-8 -*-

# test_spinnaker_stdp_window.py

# This file is part of NEST.

# Copyright (C) 2004 The NEST Initiative

# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.

# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


import os
import numpy as np
import pytest
import matplotlib.pyplot as plt
import csv
import pyNN.spiNNaker as p
from pyNN.utility.plotting import Figure, Panel

from pynestml.frontend.pynestml_frontend import generate_spinnaker_target
from pyNN.random import RandomDistribution

from datetime import datetime

from analyze_sample_profile import analyze_latest_profiles


# import models
from python_models8.neuron.builds.iaf_psc_exp_neuron_nestml import iaf_psc_exp_neuron_nestml

# get results before test after plot codes  first plot and then test
# next stdp synapses instead of static ones in here
# try built in neuron with nestml synapse ( stdp synapse)
# dont take too much time
# important parameters learning rate


def compute_cv(spike_train):
    """
    Compute the coefficient of variation (CV) for a single spike train.

    Parameters:
    spike_train (list or numpy array): Timestamps of spikes in the spike train.

    Returns:
    float: Coefficient of variation (CV) of the inter-spike intervals.
    """
    # Calculate inter-spike intervals (ISI)
    if len(spike_train) < 3:
        return np.nan  

    isi = np.diff(spike_train)

    # Calculate mean and standard deviation of ISI
    mean_isi = np.mean(isi)

    if mean_isi == 0:
        return np.nan  # Avoid division by zero

    std_isi = np.std(isi)

    # Calculate coefficient of variation
    cv = std_isi / mean_isi

    return cv


def compute_cv_for_neurons(spike_trains):
    """
    Compute the average coefficient of variation (CV) for a population of neurons.
    """
    cvs = []

    for spike_train in spike_trains:

        cv = compute_cv(spike_train)

        if not np.isnan(cv):
            cvs.append(cv)

    if not cvs:
        return np.nan  

    return np.mean(cvs)


def compute_average_firing_rate(spike_trains, n_neurons, t_sim):
    """
    Compute the population-average firing rate in Hz.
    """
    # Calculate the number of spikes
    total_spikes = sum(
        len(spike_train)
        for spike_train in spike_trains
    )

    # Convert ms to seconds
    sim_time_seconds = t_sim / 1000.0

    # Calculate average firing rate
    avg_firing_rate = total_spikes / (n_neurons * sim_time_seconds)

    return avg_firing_rate


def compare_results(reference, nestml):
    """
    Compare NESTML results with the reference implementation.
    """

    exc_relative_difference = (abs(nestml["exc_firing_rate"] - reference["exc_firing_rate"]) / reference["exc_firing_rate"])

    inh_relative_difference = (abs(nestml["inh_firing_rate"] - reference["inh_firing_rate"]) / reference["inh_firing_rate"])

    cv_absolute_difference = abs(nestml["cv"] - reference["cv"])

    return {
        "exc_relative_difference": exc_relative_difference,
        "inh_relative_difference": inh_relative_difference,
        "cv_absolute_difference": cv_absolute_difference
        }



def save_results_to_csv(reference, nestml, comparison):

    filename = "balanced_networks_results.csv"
    file_exists = os.path.isfile(filename)

    row = {
        "timestamp": reference["timestamp"],

        "N": reference["n_neurons"],
        "g": reference["g"],
        "neurons_per_core": reference["neurons_per_core"],
        "p_conn": reference["p_conn"],
        "rate_ext_input": reference["rate_ext_input"],

        "reference_exc_firing_rate": reference["exc_firing_rate"],
        "reference_inh_firing_rate": reference["inh_firing_rate"],
        "reference_cv": reference["cv"],

        "nestml_exc_firing_rate": nestml["exc_firing_rate"],
        "nestml_inh_firing_rate": nestml["inh_firing_rate"],
        "nestml_cv": nestml["cv"],

        "exc_relative_difference": comparison["exc_relative_difference"],
        "inh_relative_difference": comparison["inh_relative_difference"],
        "cv_absolute_difference": comparison["cv_absolute_difference"]
    }

    with open(filename, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=row.keys())

        if not file_exists: 
            writer.writeheader()

        writer.writerow(row)



def plot_membrane_potential(results):
    """
    Plot the membrane potential of a neuron and save the figure.
    """
    fig, ax = plt.subplots(nrows=1, dpi=300)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Membrane potential [mV]")
    ax.set_title(f"\nBalanced Network - {results['implementation']} | N={results['n_neurons']}, g={results['g']}, neurons/core={results['neurons_per_core']} \n"
                 f"p_conn={results['p_conn']}, input={results['rate_ext_input']} Hz |\n")

    ax.plot(results['v_neuron'].segments[0].filter(name=results['membranePot'])[0])

    plt.savefig(
        f"spinnaker_balanced_network_V_m_"
        f"{results['timestamp']}.png"
    )
    plt.close("all")

def plot_raster(results):
    """
    Plot the raster of spikes for excitatory and inhibitory populations and save the figure.
    """

    Figure(Panel(results['pop_exc_spikes'].segments[0].spiketrains,
                 xlabel="Time/ms",
                 xticks=True,
                 yticks=True,
                 markersize=5,
                 xlim=(0, results["t_sim"])),
               # raster plot of the presynaptic neuron spike times
           Panel(results['pop_inh_spikes'].segments[0].spiketrains,
                 xlabel="Time/ms",
                 xticks=True,
                 yticks=True,
                 markersize=5,
                 xlim=(0, results["t_sim"])),

                title=(
                    f"Balanced Network - {results['implementation']} | N={results['n_neurons']}, g={results['g']}, "
                    f"neurons/core={results['neurons_per_core']}\n"
                    f"p_conn={results['p_conn']}, input={results['rate_ext_input']} Hz | "
                    f"CV={results['cv']:.3f}, "
                    f"Exc={results['exc_firing_rate']:.2f} Hz, "
                    f"Inh={results['inh_firing_rate']:.2f} Hz \n"
                )
            )

    plt.savefig(
        f"balanced_network_N{results['n_neurons']}_g{results['g']}_core{results['neurons_per_core']}_"
        f"implementation{results['implementation']}_{results['timestamp']}.png"
    )

    plt.savefig(
        f"balanced_network_N{results['n_neurons']}_g{results['g']}_core{results['neurons_per_core']}_"
        f"implementation{results['implementation']}_{results['timestamp']}.pdf"
    )



class TestSpiNNakerBalancedNetwork:
    """SpiNNaker code generation tests"""

    @pytest.fixture(autouse=True,
                    scope="module")
    def generate_code(self):

        files = [os.path.join("models", "neurons", "iaf_psc_exp_neuron.nestml")]
        input_path = [os.path.realpath(os.path.join(os.path.dirname(__file__), os.path.join(
             os.pardir, os.pardir, s))) for s in files]

        generate_spinnaker_target(input_path,
                                  target_path="spinnaker-target",
                                  install_path="spinnaker-install",
                                  logging_level="DEBUG",
                                  module_name="nestmlmodule",
                                  suffix="_nestml")


    def run_balanced_network(self, use_nestml):

        t_sim = 1000    # total time to simulator for [ms]
        p_conn = .1    # connection probability
        rate_ext_input = 50.    # external input rate (eta parameter) [s⁻¹]
        n_neurons = 80
        n_exc = int(round(n_neurons * 0.8))
        n_inh = int(round(n_neurons * 0.2))
        g = 10.    # the ratio between excitation and inhibition
                # try -10 for asynchronous irregular activity. Try -1 for population-wide activity bursts
        neurons_per_core=16


        implementation = ("nestml" if use_nestml else "reference")

        if use_nestml:

            neuron_model = iaf_psc_exp_neuron_nestml()

            receptor_name_exc = "exc_spikes"
            receptor_name_inh = "inh_spikes"
            membranePot = "V_m"

            neuron_parameters = {
                "C_m": 250,
                "tau_m": 10,
                "tau_syn_inh": 2,
                "tau_syn_exc": 2,
                "refr_T": 2,
                "E_L": -70,
                "V_reset": -70,
                "V_th": -55
            }


        else:

            neuron_model = p.IF_curr_exp

            receptor_name_exc = "excitatory"
            receptor_name_inh = "inhibitory"
            membranePot = "v"

            neuron_parameters = {
                "tau_m": 10,
                "cm": 1E-3 * 250,
                "v_rest": -70,
                "v_reset": -70,
                "v_thresh": -55,
                "tau_syn_E": 2,
                "tau_syn_I": 2,
                "tau_refrac": 2
            }


        weight_exc = 1E3 * 0.5
        weight_inh = -g * weight_exc
        weight_input = 1E3

        if not use_nestml:
            weight_exc *= 1E-3
            weight_inh *= 1E-3
            weight_input *= 1E-3



        p.setup(timestep=1.0)

        p.set_number_of_neurons_per_core(neuron_model, neurons_per_core)

        p.set_number_of_neurons_per_core( p.SpikeSourcePoisson, 4)



        # External input
        pop_input = p.Population(100, p.SpikeSourcePoisson(rate=0.0),
                                additional_parameters={
                                    "max_rate": 50.0,
                                    "seed": 0},
                                label="Input")

        
        # excitatory and inhibitory populations

        pop_exc = p.Population(n_exc, neuron_model, label="Excitatory", seed=1)

        pop_inh = p.Population(n_inh, neuron_model, label="Inhibitory", seed=2)
            
        pop_exc.set(**neuron_parameters)
        pop_inh.set(**neuron_parameters)


        # external stimulus to exc and inh populations

        stim_exc = p.Population(
            n_exc, p.SpikeSourcePoisson(rate=rate_ext_input), label="Stim_Exc",
            additional_parameters={"seed": 3})
        stim_inh = p.Population(
            n_inh, p.SpikeSourcePoisson(rate=rate_ext_input), label="Stim_Inh",
            additional_parameters={"seed": 4})

        p.Projection(stim_exc, pop_exc, p.OneToOneConnector(), p.StaticSynapse(weight=weight_input, delay=1.), receptor_type=receptor_name_exc)
        p.Projection(stim_inh, pop_inh, p.OneToOneConnector(), p.StaticSynapse(weight=weight_input, delay=1.), receptor_type=receptor_name_exc)


        # Exc and Inh Connections

        delays_exc = RandomDistribution("normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=1.6)
        weights_exc = RandomDistribution("normal_clipped", mu=weight_exc, sigma=0.1, low=0, high=np.inf)
        conn_exc = p.FixedProbabilityConnector(p_conn)
        synapse_exc = p.StaticSynapse(weight=weights_exc, delay=delays_exc)

        delays_inh = RandomDistribution("normal_clipped", mu=0.75, sigma=0.375, low=1.0, high=1.6)
        weights_inh = RandomDistribution("normal_clipped", mu=weight_inh, sigma=0.1, low=-np.inf, high=0)
        conn_inh = p.FixedProbabilityConnector(p_conn)
        synapse_inh = p.StaticSynapse(weight=weights_inh, delay=delays_inh)

        # Recurrent projections

        p.Projection(pop_exc, pop_exc, conn_exc, synapse_exc, receptor_type=receptor_name_exc)
        p.Projection(pop_exc, pop_inh, conn_exc, synapse_exc, receptor_type=receptor_name_exc)
        p.Projection(pop_inh, pop_inh, conn_inh, synapse_inh, receptor_type=receptor_name_inh)
        p.Projection(pop_inh, pop_exc, conn_inh, synapse_inh, receptor_type=receptor_name_inh)

        # Initial membrane potentials
        if use_nestml:
            pop_exc.initialize(V_m=RandomDistribution("uniform", low=-65.0, high=-55.0))
            pop_inh.initialize(V_m=RandomDistribution("uniform", low=-65.0, high=-55.0))
        else:
            pop_exc.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))
            pop_inh.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))

        # Recording
        pop_exc[:5].record([membranePot])    # record only from the first 5 neurons
        pop_inh.record("spikes")
        pop_exc.record("spikes")

        pop_input.set(rate=rate_ext_input)

        p.run(t_sim)
        

        pop_exc_spikes = pop_exc.get_data("spikes")
        pop_inh_spikes = pop_inh.get_data("spikes")
        v_neuron = pop_exc.get_data(membranePot)

        exc_spike_trains = (pop_exc_spikes.segments[0].spiketrains)
        inh_spike_trains = (pop_inh_spikes.segments[0].spiketrains)



        exc_firing_rate = compute_average_firing_rate(exc_spike_trains, n_exc, t_sim)
        inh_firing_rate = compute_average_firing_rate(inh_spike_trains, n_inh, t_sim)

        cv = compute_cv_for_neurons(exc_spike_trains)



        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        p.end()

        print(f"Finished {implementation} simulation.")

        return {
            "implementation": implementation,
            "timestamp": timestamp,

            "t_sim": t_sim,

            "n_neurons": n_neurons,
            "n_exc": n_exc,
            "n_inh": n_inh,

            "g": g,
            "p_conn": p_conn,
            "rate_ext_input": rate_ext_input,
            "neurons_per_core": neurons_per_core,

            "exc_firing_rate": exc_firing_rate,
            "inh_firing_rate": inh_firing_rate,
            "cv": cv,

            "pop_exc_spikes": pop_exc_spikes,
            "pop_inh_spikes": pop_inh_spikes,
            "v_neuron": v_neuron,
            "membranePot": membranePot
        }



    def run_experiment(self):
        """
        Run reference and NESTML, compare their results,
        save results and generate plots.
        """

        reference = self.run_balanced_network(False)
        nestml = self.run_balanced_network(True)

        comparison = compare_results(reference, nestml)

        analyze_latest_profiles()

        # Print results

        print("\n===================================")
        print("REFERENCE")
        print("===================================")

        print(f"Exc. firing rate = {reference['exc_firing_rate']:.3f} Hz")
        print(f"Inh. firing rate = {reference['inh_firing_rate']:.3f} Hz")
        print(f"CV = {reference['cv']:.3f}")


        print("\n===================================")
        print("NESTML")
        print("===================================")

        print(f"Exc. firing rate = {nestml['exc_firing_rate']:.3f} Hz")
        print(f"Inh. firing rate = {nestml['inh_firing_rate']:.3f} Hz")
        print(f"CV = {nestml['cv']:.3f}")


        print("\n===================================")
        print("REFERENCE vs NESTML")
        print("===================================")

        print(f"Exc. relative difference = {comparison['exc_relative_difference']:.2%}")

        print(f"Inh. relative difference = {comparison['inh_relative_difference']:.2%}")

        print(f"CV absolute difference = {comparison['cv_absolute_difference']:.3f}")


        # Save results
        save_results_to_csv(reference, nestml, comparison)


        # Generate plots
        plot_membrane_potential(reference)
        plot_raster(reference)

        plot_membrane_potential(nestml)
        plot_raster(nestml)


        return reference, nestml, comparison



    # Pytest

    def test_spinnaker_balanced_network(self):

        reference, nestml, comparison = self.run_experiment()

        # Basic 
        for result in (reference, nestml):
            assert np.isfinite(result["exc_firing_rate"])
            assert np.isfinite(result["inh_firing_rate"])
            assert np.isfinite(result["cv"])
            assert result["exc_firing_rate"] > 0.0
            assert result["inh_firing_rate"] > 0.0

        # Reference vs NESTML
        exc_rate_tolerance = 0.40
        inh_rate_tolerance = 0.40
        cv_tolerance = 0.20

        assert comparison["exc_relative_difference"] < exc_rate_tolerance
        assert comparison["inh_relative_difference"] < inh_rate_tolerance
        assert comparison["cv_absolute_difference"] < cv_tolerance

