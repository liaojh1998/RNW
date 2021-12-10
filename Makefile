
parse_test_img:
	mkdir -p data/parsed_nuscene/mini/test/color
	ln -s /home/amrl_user/tongrui/RNW/data/nuscene/mini/samples data/parsed_nuscene/mini/test/color/samples
	ln -s /home/amrl_user/tongrui/RNW/data/nuscene/mini/sweeps data/parsed_nuscene/mini/test/color/sweeps
parse_data:
	python3 parse_data/parse_nuscene.py

test_ns:
	python3 test_nuscenes_disp.py night rnw_ns checkpoints/rnw_ns/checkpoint_epoch\=9.ckpt 

clean:
	rm -r data/parsed_nuscene/mini/sequences/*

clean_test:
	rm -r data/parsed_nuscene/mini/test/color/*

test_day:
	python3 test_nuscenes_disp.py day mono2_ns_day checkpoints/mono2_ns_day/checkpoint_epoch\=11.ckpt

test_day2:
	python3 test_nuscenes_disp.py day mono2_ns_day /robodata/tongrui/RNW/check_backup/mono2_ns_day/checkpoint_epoch\=99.ckpt

run_superpoint:
	python3 SuperPointPretrainedNetwork/demo_superpoint.py\
		data/parsed_nuscene/mini/sequences/2fc3753772e241f2ab2cd16a784cc680 \
		--weights_path SuperPointPretrainedNetwork/superpoint_v1.pth \
		--cuda \
		--save_matches \
