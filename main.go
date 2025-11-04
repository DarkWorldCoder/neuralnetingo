package main

import (
	"fmt"

	"github.com/DarkWorldCoder/neuralnetingo/dataset"
	"github.com/DarkWorldCoder/neuralnetingo/layer"
)

func RunModel() {
	layer1 := layer.Layer{}
	layer1.Initialization(2, 3)

	X, _ := dataset.SpiralData(100, 3)

	layer1.Forward(layer.ConvertToMatDense(X))
	fmt.Print(layer1.Output)
	dataset.PlotScatterData(X, 3)

}

func main() {

	RunModel()

}
