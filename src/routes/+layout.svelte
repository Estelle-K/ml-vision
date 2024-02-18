<script>
	import Menu from '$lib/components/menu/Menu.svelte';
	import Footer from '$lib/components/footer/Footer.svelte';
	import '../app.css';
</script>

<div class="flex flex-col min-h-screen">
	<Menu />

	<main
		class="flex flex-col flex-1 p-4 w-full box-border max-w-5xl mt-0 mb-0 ml-auto mr-auto"
	>
		<slot />
	</main>

	<Footer />
</div>
