from setuptools import setup

setup(name='lipnet',
    version='0.1.6',
    description='End-to-end sentence-level lipreading',
    url='http://github.com/rizkiarm/LipNet',
    author='Muhammad Rizki A.R.M',
    author_email='rizki@rizkiarm.com',
    license='MIT',
    packages=['lipnet'],
    zip_safe=False,
	install_requires=[
        'editdistance==0.6.0',
		'h5py==2.10.0',
		'matplotlib==3.5.1',
		'numpy==1.19.5',
		'python-dateutil==2.8.2',
		'scipy==1.1.0',
		'Pillow==9.0.1',
		'tensorflow-gpu==2.4.0',
		'Theano==1.0.5',
        'nltk==3.6.7',
        'sk-video==1.1.10',
        'dlib==19.23.0'
    ])
