using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLP
{
    [Serializable]
    public class HiddenLayer : Layer
    {
        private HiddenLayer(int numberOfNeurons, double probabilityOfRandom, Random rand) :
            base(numberOfNeurons, probabilityOfRandom, rand)
        {
        }
        public static HiddenLayer BuildHiddenLayer(IWeightBuilder weightBuilder,
            Layer previousLayer, int numberOfNeurons, double probabilityOfDropout, Random random)
        {
            HiddenLayer toReturn = new HiddenLayer(numberOfNeurons, probabilityOfDropout, random);

            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(Neuron.BuildNeuron(weightBuilder, previousLayer));
            }

            return toReturn;
        }
    }
}
