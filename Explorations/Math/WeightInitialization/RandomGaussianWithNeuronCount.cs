using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Math.WeightInitialization
{
    public class RandomGaussianWithNeuronCount : IWeightBuilder
    {
        private Math.RandomNormal _rand;

        public RandomGaussianWithNeuronCount(int numberOfWeights, double mean, Random random)
        {
            _rand = new RandomNormal(mean, 1 / System.Math.Sqrt(numberOfWeights));
        }

        public double BuildWeight()
        {
            return _rand.Next(); 
        }
    }
}
