package main

import "fmt"

type Neuron struct {
	weights []float64
	bias    float64
}

func main() {
	inputs := [4]float64{1, 2, 3, 2.5}

	n1 := Neuron{
		weights: []float64{0.2, 0.8, -0.5, 1.0},
		bias:    2.0,
	}
	n2 := Neuron{
		weights: []float64{0.5, -0.91, 0.26, -0.5},
		bias:    3,
	}
	n3 := Neuron{
		weights: []float64{-0.26, -0.27, 0.17, 0.87},
		bias:    0.5,
	}
	neurons := []Neuron{n1, n2, n3}
	var output []float64
	for _, neuron := range neurons {
		var neuron_output float64
		for i, weight := range neuron.weights {
			neuron_output += inputs[i] * weight
		}
		neuron_output += neuron.bias
		output = append(output, neuron_output)
	}
	fmt.Println("Output:", output)
	// fmt.Println(inputs, weights)
}
