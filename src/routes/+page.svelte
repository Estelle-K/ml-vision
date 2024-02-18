<script lang="ts">
	import * as tf from '@tensorflow/tfjs';
	import * as mobilenet from '@tensorflow-models/mobilenet';
	import {
		Label,
		Fileupload,
		Heading,
		List,
		Li,
		Card,
		Spinner
	} from 'flowbite-svelte';

	let fileuploadprops = {
		id: 'inputImage'
	};

	let predictionsResult: Prediction[] = [];
	let imgDisplay: string;
	let files: FileList;
	let imgtoprocess: HTMLImageElement;
	let model: mobilenet.MobileNet;
	let predictionsLoading = false;

	$: if (files) {
		imgDisplay = URL.createObjectURL(files[0]);
	}

	interface Prediction {
		className: string;
		probability: number;
	}

	const loadModel = async () => {
		const version = 2;
		const alpha = 0.5;
		model = await mobilenet.load({ version, alpha });
		console.log('Model loaded successfully');
		return model;
	};

	const preprocessImage = (imageElement: HTMLImageElement) => {
		const imageTensor = tf.browser.fromPixels(imageElement);
		const resizedImageTensor = tf.image.resizeBilinear(
			imageTensor,
			[224, 224]
		);
		// const normalizedImageTensor = resizedImageTensor.div(255.0);
		return resizedImageTensor;
	};

	const classifyImage = async (
		model: mobilenet.MobileNet,
		preprocessedImage: tf.Tensor3D
	) => {
		const predictions = await model.classify(preprocessedImage);
		return predictions;
	};

	const predictionImage = async (imgDisplay: HTMLImageElement) => {
		predictionsLoading = true;
		const model = await loadModel();
		const preprocessedImage = preprocessImage(imgDisplay);
		predictionsResult = await classifyImage(model, preprocessedImage);
		predictionsLoading = false;
	};

	const handleImageChange = () => {
		predictionImage(imgtoprocess);
		URL.revokeObjectURL(imgDisplay);
	};
</script>

<svelte:head>
	<title>ML Vision</title>
	<meta
		name="description"
		content="A simple ML Vision app"
	/>
</svelte:head>

<section class="flex flex-col gap-12">
	<Heading tag="h1">ML Vision with MobileNet images Classifier</Heading>
	<div class="ml-auto mr-auto block w-1/2">
		<Label class="pb-2 text-lg">Select an image to classify</Label>
		<Fileupload
			{...fileuploadprops}
			bind:files
			on:change={handleImageChange}
		/>
	</div>
	<Card
		size="md"
		padding="sm"
		class="ml-auto mr-auto"
	>
		<img
			hidden={!imgDisplay}
			src={imgDisplay}
			alt=""
			bind:this={imgtoprocess}
		/>
		<div class="py-4">
			<Heading tag="h3">Predictions:</Heading>
			<List
				position="outside"
				class="p-4"
			>
				{#if predictionsLoading}
					<span class="flex flex-col items-center"
						><Spinner color="blue" /></span
					>
				{:else}
					{#each predictionsResult as prediction}
						<Li
							>{prediction.className}: {Math.round(
								prediction.probability * 100
							)}%</Li
						>
					{/each}
				{/if}
			</List>
		</div>
	</Card>
</section>
