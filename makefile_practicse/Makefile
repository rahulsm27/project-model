
WHAT_TO_PRINT =  "Rahul"

print-hello-world: # simple target
	@echo "Hello-world"
#CLI to run above make print-hello-world


print-hello-world-again:print-hello-world # depenency
	@echo  "Hello-world-again "




print-macro:# depenency
	@echo  "Hello $(WHAT_TO_PRINT)"

#CLI make print-macro



# Store output of shell command like ls
LS_OUTPUT = $(shell ls) # declare command like this

print-ls-output:
	@echo $(LS_OUTPUT)


ifeq ($(WHAT_TO_PRINT),'Rahul')
	print_this = 'check'
else
	print_this = 'test'
endif

print-if-else:
	@echo $(print_this)