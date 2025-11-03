package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Neuron struct {
	weights mat.Dense
	bias    mat.Dense
}

type Layer struct {
	neurons []Neuron
}

func RunModel() {
	inputs := mat.NewDense(3, 4, []float64{1, 2, 3, 2.5, 2.0, 5.0, -1.0, 2.0, -1.5, 2.7, 3.3, -0.8})

	n1 := Neuron{
		weights: *mat.NewDense(1, 4, []float64{0.2, 0.8, -0.5, 1.0}),
		bias:    *mat.NewDense(1, 1, []float64{2.0}),
	}
	n2 := Neuron{
		weights: *mat.NewDense(1, 4, []float64{0.5, -0.91, 0.26, -0.5}),
		bias:    *mat.NewDense(1, 1, []float64{3.0}),
	}
	n3 := Neuron{
		weights: *mat.NewDense(1, 4, []float64{-0.26, -0.27, 0.17, 0.87}),
		bias:    *mat.NewDense(1, 1, []float64{0.5}),
	}

	weights2 := []*mat.Dense{
		mat.NewDense(1, 3, []float64{0.1, -0.14, 0.5}),
		mat.NewDense(1, 3, []float64{-0.5, 0.12, -0.33}),
		mat.NewDense(1, 3, []float64{-0.44, 0.73, -0.13}),
	}
	biases2 := []*mat.Dense{
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{-0.5}),
	}
	layer1 := Layer{
		neurons: []Neuron{n1, n2, n3},
	}
	layer2 := Layer{}
	for i := 0; i < 3; i++ {
		layer2.neurons = append(layer2.neurons, Neuron{
			weights: *weights2[i],
			bias:    *biases2[i],
		})
	}

	outputs1 := mat.NewDense(3, 3, nil)
	for i, neuron := range layer1.neurons {
		var result mat.Dense
		result.Mul(inputs, neuron.weights.T())
		// Broadcast bias to match result dimensions
		rr, rc := result.Dims()
		biasValue := neuron.bias.At(0, 0)
		biasData := make([]float64, rr*rc)
		for j := range biasData {
			biasData[j] = biasValue
		}
		biasBroadcast := mat.NewDense(rr, rc, biasData)
		result.Add(&result, biasBroadcast)
		outputs1.SetCol(i, mat.Col(nil, 0, &result))
	}

	outputs2 := mat.NewDense(3, 3, nil)
	for i, neuron := range layer2.neurons {
		var result mat.Dense
		result.Mul(outputs1, neuron.weights.T())
		// Broadcast bias to match result dimensions
		rr, rc := result.Dims()
		biasValue := neuron.bias.At(0, 0)
		biasData := make([]float64, rr*rc)
		for j := range biasData {
			biasData[j] = biasValue
		}
		biasBroadcast := mat.NewDense(rr, rc, biasData)
		result.Add(&result, biasBroadcast)
		outputs2.SetCol(i, mat.Col(nil, 0, &result))
	}

	fmt.Println("Final outputs:")
	fmt.Printf("%v\n", mat.Formatted(outputs2, mat.Prefix(""), mat.Excerpt(0)))
}
func main() {

	RunModel()

}
