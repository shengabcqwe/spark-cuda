package shenpeng

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

object SparkCuda{
     
 
     def main(args:Array[String]){
     val conf = new SparkConf().setAppName("PI").setMaster("local")
     val sc=new SparkContext(conf)
     val n=100000
     val jc=new JCudaAdd()
     val pi=sc.parallelize(1 to 4).map(i=>jc.getPi(i)).reduce(_+_)/n
     println("Pi is approximately :"+ pi)
     sc.stop()
   }  
}
