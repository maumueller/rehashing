DS="shuttle covtype"

for ds in $DS; do
    bin/hbe_benchmark resources/data/$ds.conf gaussian | tee $ds_$(date +"%m_%d_%Y").log
done
