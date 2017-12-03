using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class InputLayer : Layer
    {
        private InputLayer(int numberOfNeurons, Random rand) :
            base(numberOfNeurons, 0, rand)
        {
        }
        public static InputLayer BuildInputLayer(IWeightBuilder weightBuilder, int numberOfNeurons, Random random)
        {
            InputLayer toReturn = new InputLayer(numberOfNeurons, random);
            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(weightBuilder, null));
            }
            return toReturn;
        }
    }
}
