package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type Neuron struct {
	weights []float64
	bias    float64
}

func RunModel() {
	inputs := [][]float64{{1, 2, 3, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}}
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
	weights := [][]float64{
		n1.weights,
		n2.weights,
		n3.weights,
	}
	biases := []float64{n1.bias, n2.bias, n3.bias}

	input_dense := mat.NewDense(len(inputs), len(inputs[0]), nil)
	for i := 0; i < len(inputs); i++ {
		for j := 0; j < len(inputs[0]); j++ {
			input_dense.Set(i, j, inputs[i][j])
		}
	}

	weight_dense := mat.NewDense(len(weights), len(weights[0]), nil)
	for i := 0; i < len(weights); i++ {
		for j := 0; j < len(weights[0]); j++ {
			weight_dense.Set(i, j, weights[i][j])
		}
	}

	var output_dense mat.Dense
	output_dense.Mul(input_dense, weight_dense.T())

	for i := 0; i < output_dense.RawMatrix().Rows; i++ {
		for j := 0; j < output_dense.RawMatrix().Cols; j++ {
			output_dense.Set(i, j, output_dense.At(i, j)+biases[j])
		}
	}

	fmt.Printf("Output Dense:\n%v\n", mat.Formatted(&output_dense))
}
func main() {

	RunModel()

}
