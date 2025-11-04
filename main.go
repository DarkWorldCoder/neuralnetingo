package main

import (
	"fmt"

	"github.com/DarkWorldCoder/neuralnetingo/activations"
	"github.com/DarkWorldCoder/neuralnetingo/dataset"
	"github.com/DarkWorldCoder/neuralnetingo/layer"
	"github.com/DarkWorldCoder/neuralnetingo/utils"
	"gonum.org/v1/gonum/mat"
)

func RunModel() {
	layer1 := layer.Layer{}
	layer1.Initialization(2, 3)

	X, _ := dataset.SpiralData(100, 3)
	layer1.Forward(utils.ConvertToMatDense(X))
	relu1 := activations.Activation_ReLU{}
	relu1.Forward(&layer1.Output)

	layer2 := layer.Layer{}
	layer2.Initialization(3, 3)
	layer2.Forward(&relu1.Output)
	softmax1 := activations.Activation_Softmax{}
	softmax1.Forward(&layer2.Output)

	fmt.Printf("Output after Softmax:\n%v\n", mat.Formatted(&softmax1.Output, mat.Prefix(" "), mat.Excerpt(0)))

}

func main() {

	RunModel()

}
