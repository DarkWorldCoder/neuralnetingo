package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Neuron struct {
	weights []float64
	bias    float64
}

func DotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("Vectors must be of the same length")
	}
	var result float64
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func RunModel() {
	inputs := []float64{1, 2, 3, 2.5}

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
		neuron_output = DotProduct(neuron.weights, inputs[:]) + neuron.bias
		output = append(output, neuron_output)
	}
	fmt.Println("Output:", output)
}
func main() {
	a := mat.NewVecDense(3, []float64{1, 2, 3})
	b := mat.NewVecDense(3, []float64{2, 3, 4}).T()

	result := mat.NewDense(1, 1, nil)
	result.Mul(b, a)

	fmt.Printf("Result:\n%v\n", mat.Formatted(result))
	// RunModel()

}
