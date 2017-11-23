using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Math;

namespace Math
{
    public class RandomNormal
    {
        private Random _rand;
        private double _mean, _stdDev;
        public RandomNormal(double mean, double stdDev)
        {
            _rand = new Random();
            _mean = mean;
            _stdDev = stdDev;
        }

        public RandomNormal(double mean, double stdDev, int seed)
        {
            _rand = new Random(seed);
            _mean = mean;
            _stdDev = stdDev;
        }

        public RandomNormal(double mean, double stdDev, Random rand)
        {
            _rand = rand;
            _mean = mean;
            _stdDev = stdDev;
        }

        public double Next()
        {
            double u1 = 1.0 - _rand.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - _rand.NextDouble();
            double randStdNormal = Sqrt(-2.0 * Log(u1)) *
                         Sin(2.0 * PI * u2); //random normal(0,1)
            return _mean + _stdDev * randStdNormal; //random normal(mean,stdDev^2)
        }
    }
}
