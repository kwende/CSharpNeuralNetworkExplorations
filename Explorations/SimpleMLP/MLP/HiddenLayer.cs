using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SimpleMLP.MLP
{
    public class HiddenLayer : Layer
    {
        private HiddenLayer() : base()
        {
        }
        public static HiddenLayer BuildHiddenLayer(Math.RandomNormal rand, Layer previousLayer, int numberOfNeurons)
        {
            HiddenLayer toReturn = new HiddenLayer();

            for (int c = 0; c < numberOfNeurons; c++)
            {
                toReturn.Neurons.Add(WeightedNeuron.BuildNeuron(rand, previousLayer));
            }

            return toReturn;
        }
    }
}
