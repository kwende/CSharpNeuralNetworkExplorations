using Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Math.WeightInitialization
{
    public class RandomGaussian : IWeightBuilder
    {
        private Math.RandomNormal _rand;

        public RandomGaussian(double mean, double stdDev, Random random)
        {
            _rand = new Math.RandomNormal(0, 1, random);
        }

        public double BuildWeight()
        {
            return _rand.Next();
        }
    }
}
