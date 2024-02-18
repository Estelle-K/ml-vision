<script lang="ts">
	import {
		Navbar,
		NavBrand,
		NavLi,
		NavUl,
		NavHamburger
	} from 'flowbite-svelte';
</script>

<Navbar
	let:hidden
	let:toggle
	class="bg-black text-white"
>
	<NavBrand href="/"></NavBrand>
	<NavHamburger
		on:click={toggle}
		class="hover:bg-[#6C70ED]"
		menuClass="outline-[#6C70ED]"
	/>
	<NavUl
		{hidden}
		ulClass="bg-black flex flex-col p-4 mt-4 md:flex-row md:space-x-8 rtl:space-x-reverse md:mt-0 md:text-sm md:font-medium"
		divClass="w-full md:block md:w-auto"
	>
		<NavLi
			href="/"
			active={true}
			class="md:text-white md:hover:text-[#6C70ED]"
			nonActiveClass="text-white">Home</NavLi
		>
	</NavUl>
</Navbar>
