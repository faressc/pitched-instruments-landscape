import lmdb
import os
import gc
try:
    from proto.meta_audio_file_pb2 import MetaAudioFile
except:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from proto.meta_audio_file_pb2 import MetaAudioFile

def strip_audiofile_from_protobuf(input_db_path: str, output_db_path: str):
    # Open the input LMDB environment
    env_in = lmdb.open(input_db_path, readonly=True, lock=False)
    
    # Create the output LMDB environment
    env_out = lmdb.open(output_db_path, map_size=env_in.info()['map_size'])
    
    index = 0
    with env_in.begin() as txn_in:
        cursor = txn_in.cursor()
        txn_out = env_out.begin(write=True)

        for key, value in cursor:
            meta_audio_file = MetaAudioFile()
            meta_audio_file.ParseFromString(value)
            
            # Strip the audio_file message
            meta_audio_file.ClearField('audio_file')
            
            # Serialize the modified protobuf
            stripped_value = meta_audio_file.SerializeToString()
            
            # Write the modified protobuf to the output database
            txn_out.put(key, stripped_value)

            del meta_audio_file, stripped_value
            index += 1
            if index % 1000 == 0:
                print(f"Processed {index} records")
                gc.collect()
                txn_out.commit()
                txn_out = env_out.begin(write=True)

        txn_out.commit()
        cursor.close()

    print(f"Processed {index} records in total")    
    env_in.close()
    env_out.close()

if __name__ == "__main__":
    input_db_path = "data/partial/train"
    output_db_path = "data/partial/train_stripped"
    
    if not os.path.exists(output_db_path):
        os.makedirs(output_db_path)
    
    strip_audiofile_from_protobuf(input_db_path, output_db_path)
    print(f"Stripped audio_file messages and saved to {output_db_path}")
