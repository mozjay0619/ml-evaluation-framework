
from future.standard_library import install_aliases
install_aliases()
from urllib.parse import urlparse, urlencode
import os
import boto3
import errno
import shutil


def s3_upload_object(file_path, s3_url, object_name):
	"""Upload a file to S3 bucket in designated key name. 

	Parameters
	----------
	file_path : String
		The path to the file, including the name of the file.
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name : String
		The desired key name to be uploaded in S3 bucket.
	"""
	
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	key = os.path.join(path[1:], object_name)
	
	s3_resource = boto3.resource('s3')
	s3_client = s3_resource.meta.client
	s3_client.upload_file(Filename=file_path, Bucket=bucket, Key=key)


def s3_download_object(dest_dir, s3_url, object_name_prefix):
	"""Download all objects with designated name prefix from S3 bucket into desired destination directory. 
	If the destination directory does not exist, create one before downloading the objects into it. 

	Parameters
	----------
	dest_dir : String
		The path to the directory into which the objects are to be stored.
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name_prefix : String
		The prefix of the object name to use to filter the download.
	"""
	
	try:
		os.makedirs(dest_dir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
	
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	prefix = os.path.join(path[1:], object_name_prefix)
	
	s3_resource = boto3.resource('s3')
	s3_bucket = s3_resource.Bucket(name=bucket)
	
	for obj in s3_bucket.objects.filter(Prefix=prefix):
		
		filename = obj.key.split('/')[-1]
		filepath = os.path.join(dest_dir, filename)
		
		s3_bucket.download_file(obj.key, filepath)


def s3_upload_zip_dir(source_dir, s3_url, object_name):
	"""First create a zipped version of the desired directory to upload, and then upload the zipped 
	directory into S3. Delete the zipped directory afterwards. 

	Parameters
	----------
	source_dir : String
		The path of the directory to be zipped and uploaded.
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name : String
		The desired name for the S3 key of the uploaded zipped directory file.
	"""
	
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	key = os.path.join(path[1:], object_name)
	
	zip_file_dir = shutil.make_archive(source_dir, 'zip', source_dir)
	
	s3_resource = boto3.resource('s3')
	s3_client = s3_resource.meta.client
	s3_client.upload_file(Filename=zip_file_dir, Bucket=bucket, Key=key)
	
	os.remove(zip_file_dir)
	

def s3_delete_object(s3_url, object_name_prefix):
	"""Delete all objects with the designated key name prefix from S3 bucket.
	
	Parameters
	----------
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name_prefix : String
		The key name prefix to filter the deletion. 
	"""
	
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	prefix = os.path.join(path[1:], object_name_prefix)
	
	s3_resource = boto3.resource('s3')
	s3_bucket = s3_resource.Bucket(name=bucket)
	
	for obj in s3_bucket.objects.filter(Prefix=prefix):
		
		s3_bucket.Object(key=obj.key).delete()


def s3_object_exists(s3_url, object_name):
	"""Return boolean value of True if the object exists in S3, False otherwise.
	
	Parameters
	----------
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name : String
		The total and exact name of the key of the object. 
	"""
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	key = os.path.join(path[1:], object_name)
	
	s3_resource = boto3.resource('s3')
	
	try:
		s3_resource.Object(bucket_name=bucket, key=key).load()
		return True
		
	except botocore.exceptions.ClientError as e:
		if e.response['Error']['Code'] == "404": 
			return False
		else:
			return False  # in case we want to handle non 404 differently in the future
		

def s3_fetch_keys(s3_url, object_name_prefix):
	"""Get all keys from S3 bucket with the given key name prefix.
	
	Parameters
	----------
	s3_url : String
		The s3 url up to and including the bucket name.
	object_name_prefix : String
		The prefix of the object name to use to filter the search.
	"""
	scheme, bucket, path, _, _, _ = urlparse(s3_url)
	prefix = os.path.join(path[1:], object_name_prefix)
	
	s3_resource = boto3.resource('s3')
	s3_client = s3_resource.meta.client
	
	kwargs = {'Bucket': bucket, 'Prefix': prefix}
	keys = []
	
	while True:
		
		resp = s3_client.list_objects_v2(**kwargs)

		if not 'Contents' in resp:
			break
		
		for obj_metadata_dict in resp['Contents']:
			
			key = obj_metadata_dict['Key']
			keys.append(key)
			
		try:
			kwargs['ContinuationToken'] = resp['NextContinuationToken']
		except KeyError:
			break
	
	return keys


if __name__ == '__main__':

	file_dir = modelfile
	s3_url = 's3://grubhub-gdp-data-transfer-dev/pickup_feasibility'
	object_name = 'model_name_XXX'

	s3_upload_object(file_dir, s3_url, object_name)


	dest_dir = os.getcwd() + '/model_dir_date-XXX'
	s3_url = 's3://grubhub-gdp-data-transfer-dev/pickup_feasibility'
	object_name_prefix = 'model_name'

	s3_download_object(dest_dir, s3_url, object_name_prefix)


	source_dir = dest_dir
	s3_url = 's3://grubhub-gdp-data-transfer-dev/pickup_feasibility'
	object_name = 'zipped_models'

	s3_upload_zip(source_dir, s3_url, object_name)


	s3_url = 's3://grubhub-gdp-data-transfer-dev/pickup_feasibility'
	object_name_prefix = 'model_name'

	s3_delete_object(s3_url, object_name_prefix)
