DS="shuttle covtype"

bash install.sh

for ds in $DS; do
    if [ ! -f "resources/data/$ds.conf" ]; then
        echo "Generating $ds"
        python3 data/preprocess_datasets.py --dataset $ds
    fi

    bin/hbe_benchmark resources/data/$ds.conf gaussian | tee ${ds}_$(date +"%m_%d_%Y").log
done
