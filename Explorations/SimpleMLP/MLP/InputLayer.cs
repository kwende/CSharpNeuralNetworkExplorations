﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    [Serializable]
    public class InputLayer : Layer
    {
        private InputLayer() : base()
        {
        }
        public static InputLayer BuildInputLayer(Math.RandomNormal rand, int numberOfNeurons)
        {
            InputLayer toReturn = new InputLayer();
            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(rand, null));
            }
            return toReturn;
        }
    }
}
